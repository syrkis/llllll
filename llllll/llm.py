from chex import dataclass
from jax import random, tree_util, lax
from functools import partial
import darkdetect
import jax.numpy as jnp
from chex import dataclass
from dataclasses import field
import jaxmarl
from typing import Tuple, List, Dict, Optional, Callable
import parabellum as pb
import numpy as np

# from plot import int_to_color
# import utils
import datetime
from copy import deepcopy
import cv2
from time import time
import os
import pickle
import multiprocessing
from enum import Enum
from PIL import Image
import matplotlib.pyplot as plt

import parabellum as pb


# ### Objectives

# +
objective_types = ["position", "eliminate"]
default_distance_threshold = 3


# ### Steps


# ### Parser


# +
class PlanParsingError(Exception):
    pass


def parse_list(txt):
    if txt[0] == "[" and txt[-1] == "]":
        if len(txt) == 2:
            return []
        for x in txt[1:-1].split(", "):
            if not x.isdigit():
                raise PlanParsingError(f'"{x}" is not a number.')
        return [int(x) for x in txt[1:-1].split(", ")]
    else:
        raise PlanParsingError(f'"{txt}" is not a list.')


def parse_coordinate(txt):
    if txt[0] == "(" and txt[-1] == ")":
        x, y = txt.split(", ")[0][1:], txt.split(", ")[1][:-1]
        if not x.isdigit():
            raise PlanParsingError(f'"{x}" is not a number.')
        if not y.isdigit():
            raise PlanParsingError(f'"{y}" is not a number.')
        return int(x), int(y)
    raise PlanParsingError(f'"{txt}" is not a tuple of integers.')


def parse_plan(plan_txt, num_allies, num_enemies, default_distance_threshold, objective_types, LLM_BTs, for_ally=True):
    plan = {}
    for step in plan_txt.split("Step ")[1:]:
        config = [line for line in step.split("\n") if line]
        if len(config) < 6:
            raise PlanParsingError(
                f"{answer} is missing mandatory parameters. There should be in order: Step ID - prerequisites - objective - [units - target position - behavior]+"
            )

        target_pos = {}

        # step id
        step_id = config[0][:-1]
        if step_id.isdigit():
            step_id = int(step_id)
        else:
            raise PlanParsingError(f'"{step_id}" is not a number.')

        # prerequisites
        if config[1][:15] != "prerequisites: ":
            raise PlanParsingError(f'"{config[1][:15]} is undefined. Should be "prerequisites: "')
        prerequisites = parse_list(config[1][15:])

        # objective
        if config[2][:11] != "objective: ":
            raise PlanParsingError(f'"{config[2][:11]} is undefined. Should be "objective: "')
        objective_type = config[2][11:].split(" ")[0]
        if objective_type == "position":
            objective = "position"
        elif objective_type == "elimination":
            targets = config[2][23:]
            if for_ally:
                targets = (
                    [num_allies + i for i in range(num_enemies)]
                    if targets == "all"
                    else [num_allies + i for i in parse_list(targets)]
                )
            else:
                targets = [i for i in range(num_allies)] if targets == "all" else [i for i in parse_list(targets)]
            objective = EliminationObjective(target_units=targets)
        else:
            raise PlanParsingError(f'"{objective_type}" is undefined. Should be in {objective_types}')

        all_units = []
        all_target_pos = {}
        all_behaviors = {}
        concerned_units_per_target = {}
        line_idx = 3
        while len(config[line_idx:]) >= 3:  # [units - target position - behavior]+
            # concerned units
            if config[line_idx][:7] != "units: ":
                raise PlanParsingError(f'"{config[line_idx][:7]}" is undefined. Should be "units: "')
            units = config[line_idx][7:]
            if units == "all":
                units = [i for i in range(num_allies)] if for_ally else [i + num_allies for i in range(num_enemies)]
            else:
                units = parse_list(units) if for_ally else [i + num_allies for i in parse_list(units)]
            for i in units:
                if i in all_units:
                    raise PlanParsingError(f"Unit {i} has several assigned behaviors in Step {step_id}.")
                else:
                    all_units.append(i)

            # target_pos
            if config[line_idx + 1][:19] != "- target position: ":
                raise PlanParsingError(f'"{config[line_idx+1][:19]}" is undefined. Should be "target position: "')
            target_pos = parse_coordinate(config[line_idx + 1][19:])
            for i in units:
                all_target_pos[i] = target_pos
            concerned_units_per_target[target_pos] = units

            # BT
            if config[line_idx + 2][:12] != "- behavior: ":
                raise PlanParsingError(f'"{config[line_idx+2][:12]}" is undefined. Should be "behavior: "')
            behavior = config[line_idx + 2][12:].split(" ")
            bt = behavior[0]
            if bt not in LLM_BTs:
                raise PlanParsingError(f'"{bt}" is undefined. Should be in {LLM_BTs.keys()}.')
            ## valid unit type
            targets = set()
            if len(behavior[1:]) > 0:
                if behavior[1] == "any":
                    targets = unit_types_set
                else:
                    for target in behavior[1:]:
                        if target:
                            if target not in unit_types:
                                raise PlanParsingError(f'"{target}" is undefined. Should be in {unit_types}.')
                            else:
                                targets.add(target)
            if len(targets) == 0:
                targets = unit_types_set
            for i in units:
                all_behaviors[i] = find_bt_key(LLM_BTs[bt], targets)
            line_idx += 3
        if len(config[line_idx:]) > 0:
            raise PlanParsingError(
                f'"{config[line_idx:]}" is not a valid units behavior assignement. Should be in the form "units:\n- target position:\n- behavior:"'
            )

        if objective == "position":
            objective = PositionObjective(
                target_position=all_target_pos, concerned_units_per_target=concerned_units_per_target
            )

        plan[step_id] = Step(
            **{
                "objective": objective,
                "prerequisites": prerequisites,
                "units": all_units,
                "target_pos": all_target_pos,
                "assigned_bt": all_behaviors,
            }
        )
    return plan


# -
