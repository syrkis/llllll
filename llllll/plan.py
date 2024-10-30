from chex import dataclass
from dataclasses import field
from llllll import bts
import numpy as np
import jax.numpy as jnp
from jax import random, tree_util, lax
from functools import partial
import jaxmarl
from typing import Tuple, List, Dict, Optional, Callable
import parabellum as pb
import numpy as np

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
from llllll import bts
import llllll as ll


# ### Objectives

# +
objective_types = ["position", "eliminate"]
default_distance_threshold = 3

true_enemies_subset = {"soldier", "sniper", "swat"}
unit_types = ["soldier", "sniper", "swat", "turret", "civilian"]
unit_types_set = {"soldier", "sniper", "swat", "turret", "civilian"}


# %% Constants
enemy_plan = ll.env.enemy_plan


ally_plan = f"""Step 0:
prerequisites: []
objective: position
units: all
- target position: {(0, 0)}
- behavior: stand
"""

# ally_plan = ll.env.winning_plan


@dataclass
class Objective:
    def check(self, game, concerned_units):  # type: ignore
        return False


@dataclass
class EliminationObjective(Objective):
    target_units: list

    def check(self, game, _):  # type: ignore
        return np.sum([game.state.unit_health[i] for i in self.target_units]) == 0


@dataclass
class PositionObjective(Objective):
    target_position: dict  # unit_id: (x, y)
    concerned_units_per_target: dict  # (x, y): [unit_ids]
    strict: bool = True

    def get_distances_to_targets(self, game, concerned_units):
        return {
            target: 1
            + np.ceil(
                np.sum(
                    [
                        game.env.unit_type_radiuses[game.state.unit_types[j]] ** 2
                        for j in self.concerned_units_per_target[target]
                        if game.state.unit_health[j] > 0
                    ]
                )
                ** 0.5
            )
            for target in self.concerned_units_per_target.keys()
        }

    def check(self, game, concerned_units):
        distances = self.get_distances_to_targets(game, concerned_units)
        if self.strict:
            return (
                np.sum(
                    [
                        jnp.linalg.norm(game.state.unit_positions[i] - jnp.array(self.target_position[i]))
                        > distances[self.target_position[i]]
                        for i in concerned_units
                        if game.state.unit_health[i] > 0
                    ]
                )
                == 0
            )
        else:
            return (
                np.sum(
                    [
                        jnp.linalg.norm(game.state.unit_positions[i] - jnp.array(self.target_position[i]))
                        <= distances[self.target_position[i]]
                        for i in concerned_units
                        if game.state.unit_health[i] > 0
                    ]
                )
                > 0
            )


@dataclass
class Step:
    objective: Objective
    prerequisites: list = field(default_factory=list)
    units: list = field(default_factory=list)  # list of units id concerned by the objectives
    target_pos: dict = field(default_factory=dict)  # assigned target position for each unit id in units
    assigned_bt: dict = field(default_factory=dict)  # assigned BT for each unit id in units
    done: bool = False

    def check_objective(self, game):
        self.done = self.objective.check(game, self.units)
        return self.done

    def check_prerequisites(self, plan):
        if self.prerequisites:
            return np.sum([not plan[i].done for i in self.prerequisites]) == 0
        else:
            return True


# ### Initialization
def initialize_plan(game):
    for step in game.ally_plan.values():
        for target in step.target_pos.values():
            compute_direction_map(game, target)
    for step in game.enemy_plan.values():
        for target in step.target_pos.values():
            compute_direction_map(game, target)


def compute_direction_map(game, target):
    target = (int(target[0]), int(target[1]))
    mask = 1 - jnp.logical_or(game.env.terrain.building, game.env.terrain.water)
    if target not in game.direction_maps:
        game.direction_maps[target] = ll.env.compute_bfs(mask, target)[1]


def reset_plan(game):
    initialize_plan(game)
    for step in game.ally_plan.values():
        step.done = False
    for step in game.enemy_plan.values():
        step.done = False


# -

# ### Execution


def execute_plan(game):
    if len(game.ally_plan) > 0:
        initialize_plan(game)


def get_current_steps(game):
    current_ally_steps = []
    for step in game.ally_plan.values():
        if not step.done and step.check_prerequisites(game.ally_plan) and not step.check_objective(game):
            current_ally_steps.append(step)
    current_enemy_steps = []
    for step in game.enemy_plan.values():
        if not step.done and step.check_prerequisites(game.enemy_plan) and not step.check_objective(game):
            current_enemy_steps.append(step)
    return current_ally_steps, current_enemy_steps


def apply_plan(game):  # changes bt and target of each agent.
    # RED FLAG: get unit actions in parallel with batchify.
    current_ally_steps, current_enemy_steps = get_current_steps(game)
    if current_ally_steps == []:
        return 1
    elif current_enemy_steps == []:
        return -1
    for unit_id in range(
        game.env.num_agents
    ):  # reset the direction map of each unit so that it stands if no direction map is given and it uses the follow_direction atomic
        game.agent_info.direction_map = game.agent_info.direction_map.at[unit_id].set(
            jnp.ones(game.env.terrain.building.shape, dtype=jnp.int32)
        )
    for current_step in current_ally_steps + current_enemy_steps:
        for unit_id in current_step.units:
            unit_name = game.env.agents[unit_id]  # maybe
            bt_name = current_step.assigned_bt[unit_id]
            game.assigned_bts[unit_name] = game.bt_name_dict[bt_name]
            if unit_id in current_step.target_pos:
                target = current_step.target_pos[unit_id]
                if target not in game.direction_maps:
                    compute_direction_map(game, target)
                game.agent_info.direction_map = game.agent_info.direction_map.at[unit_id].set(
                    game.direction_maps[target]
                )
    return 0


# ### Ask LLM


def parse_plan(
    plan_txt,
    num_allies,
    num_enemies,
    default_distance_threshold,
    objective_types,
    LLM_BTs,
    bts_bank_variant_subsets,
    for_ally=True,
):
    plan = {}
    for step in plan_txt.split("Step ")[1:]:
        config = [line for line in step.split("\n") if line]
        if len(config) < 6:
            raise PlanParsingError(
                f"{plan_txt} is missing mandatory parameters. There should be in order: Step ID - prerequisites - objective - [units - target position - behavior]+"
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
                all_behaviors[i] = bts.find_bt_key(LLM_BTs[bt], targets, bts_bank_variant_subsets)
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
