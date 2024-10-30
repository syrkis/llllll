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
from openai import OpenAI

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
from llllll import bts, plan


# read .env for OPENAI_API_KEY


# ### Objectives

# +

# ### Steps


# ### Parser
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)


def handle_LLM_plan(game, answer_txt):
    if (
        "BEGIN PLAN" in answer_txt
        and "END PLAN" in answer_txt
        and answer_txt.count("BEGIN PLAN") == answer_txt.count("END PLAN")
    ):
        offset = 0
        nl_txt = ""
        # plans = []
        # reset_selected_plan(game)
        for i in range(answer_txt.count("BEGIN PLAN")):
            i_start = answer_txt.index("BEGIN PLAN", offset)
            i_end = answer_txt.index("END PLAN", i_start)
            nl_txt = answer_txt[offset : i_start - 1]
            name = nl_txt.split("\n")[-1].split(":")[0].replace("*", "")
            offset = i_end + 9
            plan_txt = answer_txt[i_start + 11 : i_end - 1]
            # game.control.llm_text.append(llm_text(nl_txt))
            ally_plan = plan.parse_plan(
                plan_txt,
                game.env.num_allies,
                game.env.num_enemies,
                3,  # why is there a 3 here?
                ["position", "eliminate"],
                bts.LLM_BTs,
                game.bts_bank_variant_subsets,
                # for_ally=False,
            )
            return ally_plan

            # game.control.plans.append(
            # {"plan": plan, "text": plan_txt, "name": name, "validity": ("\nValid plan!", "forestgreen")}
            # )
            # except PlanParsingError as err:
            # game.control.plans.append(
            # {
            # "plan": [],
            # "text": plan_txt,
            # "name": name,
            # "validity": ("\nInvalid plan!\n" + str(err), "crimson"),
            # }
            # )
        # game.control.llm_text.append(llm_text(answer_txt[offset:]))
        # if len(game.control.plans) == 1:
        # game_select_plan(game, 0)
    # else:
    # game.control.llm_text.append(llm_text(answer_txt + "\n"))


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


def discuss_plan_with_LLM(config, prompt, conversation, game_info, selected_plan=None, markers=""):
    messages = [system_prompt(config["instruction"])]
    # if conversation:
    # messages += conversation
    messages += game_info
    messages.append(system_prompt(config["instruction"]))
    # if markers:
    # messages.append(user_prompt(markers))
    # if selected_plan is not None:
    # messages.append(system_prompt("The player is currently looking at the following plan: " + selected_plan))
    messages.append(user_prompt(prompt))
    answer = ask_llm(config["model"], messages, temperature=config["temperature"], max_tokens=config["max_tokens"])[
        "answer"
    ]
    # answer = pre_computed_answer
    # answer = "blah balh blah we dont care"  # \n".join([x.split("#")[0].rstrip().lstrip() for x in answer.split("\n")])
    # if markers:
    # conversation.append(user_prompt(markers))
    conversation.append(user_prompt(prompt))
    conversation.append(assistant_prompt(answer))
    return {"answer": answer, "conversation": conversation}


def user_prompt(prompt):
    return {"role": "user", "content": prompt}


def system_prompt(prompt):
    return {"role": "system", "content": prompt}


def assistant_prompt(prompt):
    return {"role": "assistant", "content": prompt}


def ask_llm(model, messages, temperature=1.0, max_tokens=500):
    # if is_ollama_llm(model):
    # foo = ask_ollama
    # elif is_ChatGPT_llm(model):
    # foo = ask_ChatGPT
    # save_folder = utils.create_save_folder(utils.ROOT_PATH + "LLM/")
    t1 = time()
    foo = ask_ChatGPT
    dic = foo(model, messages, temperature, max_tokens)
    dic["messages"] = deepcopy(messages)
    dic["messages"].append(assistant_prompt(dic["answer"]))
    dic["walltime"] = time() - t1
    dic["model"] = model
    dic["temperature"] = temperature
    # utils.save_pickle(os.path.join(save_folder, "evaluate_instructions.pk"), dic)
    return dic


def ask_ChatGPT(model, messages, temperature, max_tokens):
    params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": 0,
    }

    result = client.chat.completions.create(**params)
    res = {
        "full": result,
        "answer": result.choices[0].message.content,
        "price": result.usage.completion_tokens * 15e-6 + result.usage.prompt_tokens * 5e-6,
        "config": params,
    }
    return res


################################################################


env_kwargs = {
    "unit_type_velocities": jnp.array([1.0, 1.0, 4.0, 0.0, 0.0, 1.0]),
    "unit_type_attacks": jnp.array([1.0, 2.0, 1.0, 10.0, 0.0, 0.0]),
    "unit_type_attack_ranges": jnp.array([3.0, 15.0, 2.0, 5.0, 0.0, 0.0]),
    "unit_type_sight_ranges": jnp.array([15.0, 15.0, 15.0, 5.0, 0.0, 10.0]),
    "unit_type_radiuses": jnp.array([1.0, 0.5, 0.75, 0.3, 0.0, 0.1]),
    "unit_type_health": jnp.array([50.0, 10.0, 30, 200.0, 0.0, 1.0]),
    "unit_type_weapon_cooldowns": jnp.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0]),
    "unit_type_names": ["soldier", "sniper", "swat", "turret", "drone", "civilian"],
    "unit_type_pushable": jnp.array([1, 1, 1.0, 0, 0, 1]),
    "reset_when_done": False,
}

unit_types_description = "\nDescription of the unit types in the game:\n"
for i in range(len(env_kwargs["unit_type_names"])):
    unit_types_description += f'{env_kwargs["unit_type_names"][i]}: Health={int(env_kwargs["unit_type_health"][i])};'
    unit_types_description += f' sight range={env_kwargs["unit_type_sight_ranges"][i]};'
    unit_types_description += f' attack range={env_kwargs["unit_type_attack_ranges"][i]};'
    unit_types_description += f' moving speed={env_kwargs["unit_type_velocities"][i]:1.2f};'
    unit_types_description += f' attack damage={env_kwargs["unit_type_attacks"][i]};'
    unit_types_description += f' attack cooldown={env_kwargs["unit_type_weapon_cooldowns"][i]}'
    unit_types_description += "\n"
# print(unit_types_description)
unit_types_description += f"""The Soldiers are strong against SWATs because they have more health.
The SWATs are strong against Snipers because they quicly go in close combat where the Snipers are weak.
The Snipers are strong against Soldiers because they can attack for a longer distance.\n"""

# ## BTs Bank

# ## Scenario

common_instruction = """Your goal is to devise a plan to win the scenario. The plan can consist of several steps and must encompass the whole scenario. You will lose if your plan is completed, but the scenario's winning condition is not met.
You can ask the AI assistant for more details about the composition of a plan. Once a plan is designed, you will see on the screen several elements of it:
- colored circle (with the plan step number on the top-right corner) symbolizing a position objective -> this objective is completed once all concerned units are inside the circle
- a colored "+" cross on enemy units (with the plan steps numbers on the top-right corner) symbolizing the enemy targets -> the objective is completed once all the targets are eliminated
- a colored "x" cross on the map (with the plan steps numbers on the top of it) symbolizing the target position of the elimination objective (where your units will converge to eliminate the enemies)
- each unit will have a list of its assigned step numbers of the plan on its top-right corner

You can also right-click on the map to set markers (symbolized by a dot and a capital letter) and right-click again on the same position (the dot) to remove it. The AI assistant can see them, so you can pinpoint important positions on the map and elaborate a plan.
"""

game_mode_human_instructions = {
    "HUMAN": """As a player, your task is to command the AI assistant through natural language prompting to write a successful plan and win the game's scenario.
    The AI assistant knows how to write a plan using information from the current situation. You can ask it questions about the game or the plan. You can also ask to modify the plan."""
}


introduction_text = common_instruction + "\n" + game_mode_human_instructions["HUMAN"]

# ## LLM instruction

image_map_instruction = f"""
You will be given an image of the current map.
In the game, there are four types of terrain:
- normal (in white): the units can cross and see through (by default, the whole map is normal)
- buildings (in light gray): the units cannot cross or see through buildings
- water (in blue): the units cannot cross over water but can see over it
- trees (in green): the units cannot see through trees but can move over them. In particular, once a unit is inside a tree area, it cannot see any other unit
The map will be enhanced with a scaffolding of coordinates (in black) to help you understand the positioning on the map.
Use those different coordinates when talking to the player.

The Soldiers are symbolized by squares.
The SWATs are symbolized by triangles.
The Snipers are symbolized by circles.
"""

text_map_instruction = f"""
You will be given a textual description of each pertinent element of the map using their name, their type of terrain, and their bounding boxes (bottom left corner - top right corner).
In the game, there are four types of terrain:
- normal: the units can corrs and see through (by default, the whole map is normal)
- buildings: the units cannot cross or see through buildings
- water: the units cannot cross over water but can see over it
- trees: the units cannot see through trees but can move over them. In particular, once a unit is inside a tree area, it cannot see any other unit

Also, for simplicity, bridges that allow crossing water terrain will be specified. They correspond to normal terrain.
You can assume that any part of the map that is not included in any of the pertinent elements has a normal type of terrain.
A common consensus is that East = Right, North = Top, West = Left, and South = Bottom. So the point (0, 0) is the bottom left corner of the map.
The x-axis increases from West to East = from Left to Right, and the y-axis increases from South to North = from Bottom to Top.

Format:
[Name]: [terrain type] at [(x, y) coordinates of the bottom left corner] - [(x, y) coordinates of the top right corner]

### For example
East forest: trees at (12, 63) - (56, 89)
North river: water at (0, 85) - (100, 90)
North river's bridge: normal at (45, 85) - (55, 90)
"""

vision = "text"  # "text" or "image" or "both"

game_description = f"""

# Map instruction

You are a game assistant that helps the player in a strategy video game.
Units also block each other movement by pushing each other.

{image_map_instruction if vision in ['image', "both"] else ""}

{text_map_instruction if vision in ["text", "both"] else ""}

## Markers

Through the discussion, the player can define markers on the map, which will be provided to you using the following format:
Markers:
A at (17, 5)
B at (6, 32)
C at (25, 28)

# Information about the units

{unit_types_description}

You will be provided with the list of units in the game with their affiliated team (Ally or Enemy), their type, their position on the map, their health, and max health if they are alive or simply dead if they had been eliminated from the game.

## For example

Units:
ally 0: type = soldier; position = (0.2, 4.2); health/max health) = 5/45
ally 1: type = sniper; position = (1.2, 0.0); health/max health) = 35/45
ally 2: dead
ally 3: dead
ally 4: type = sniper; position = (1.2, 0.0); health/max health) = 45/45
enemy 0: type = sniper; position = (26.5, 27.2); health/max health) = 45/45
enemy 1: dead
enemy 2: type = sniper; position = (30.0, 28.1); health/max health) = 25/45
enemy 3: type = soldier; position = (29.3, 26.2); health/max health) = 1/45
enemy 4: type = soldier; position = (30.4, 25.5); health/max health) = 5/45

"""

# ### Assistant AI

game_mode_llm_instructions = {
    "HUMAN": """Your task is to discuss with a player to come up with a plan to win the game's scenario (which will be provided later).
    Work with the player by giving feedback about their propositions, asking questions to clarify, obtain more details, or decide between different propositions.
    The player can also ask you questions.""",
}

# + active=""
# to add in the list of available behaviors
# - ignore_enemies: the unit simply moves toward the target position while ignoring the enemies. This is a risky behavior as the enemy can easily eliminate the unit.
# -

instruction = f"""
{game_description}

{game_mode_llm_instructions["HUMAN"]}

# How you should handle the player's prompt

Most of the time, the user will only ask you for one big plan. If you haven't proposed one yet, do so.
If you already have a shared conversation and the user asks you to modify or add steps to the plan, take care to build upon the already selected plan.

If the player asks you to propose different plans, present an enumeration of five different strategies following the following format:
ID. NAME: a once-sentence high-level description (be concise and clear so the player can quickly assess the strategies)
BEGIN PLAN
[a detailed plan whose syntax is explained below. START EACH LINE OF THE PLAN WITHOUT SPACES!]
END PLAN

## For Example:
1. Offensive: We rush all units toward the enemies to outpower them.
BEGIN PLAN
[a detailed plan whose syntax is explained below]
END PLAN

2. Stealth: We flank the enemies by moving undercover through the trees.
BEGIN PLAN
[a detailed plan whose syntax is explained below]
END PLAN

3. Defensive advance: We move methodically to maximize our defensive abilities while drawing the enemies toward us.
BEGIN PLAN
[a detailed plan whose syntax is explained below]
END PLAN

For the short description, refer to specific units using their integer IDs.

The player may ask you to propose other strategies or strategies similar to one of those you propose.
In that case, propose another enumeration.
If the player asks you to change one of the plans or to design a plan given directions, simply propose one plan using the same format but with only one item.

IMPORTANT: most of the time, the user will only ask you for one big plan with possibly several steps and/or different groups for each step; in that case, only propose one plan.

# Syntax for a detailed plan

A detailed plan is a set of steps to achieve in a given order until all the steps are completed.
You must provide the steps of the plan between the two keywords "BEGIN PLAN" and "END PLAN"

One step comprises:
- a numeral ID
- a list of prerequisite steps that need to be completed before the step is rolled out
- an objective
A succession (at least one) of groups of units:
- the unit IDs of the group
- target position on the map for the units to go to if there are no enemies in sight
- a behavior for the units if there are enemies in sight

The list of prerequisites corresponds to the list of steps' IDs that must be completed before trying to achieve the objective.

There are two kinds of objectives:
- elimination objective: where the objective is completed when all the targets are eliminated. In that case, you must provide a list of enemies' IDs or the keyword "all" if all enemies are targets.
- position objective: where the objective is completed when all the concerned units are close to their target position.

Position objectives are a good way to move your units, but if the end objective is to eliminate enemies, it can be more straightforward to directly set an elimination objective and use the target position to move the units.
The concerned units are either the keyword "all" (if all the allies units are concerned) or a list of integers corresponding to the unit IDs.

The behavior corresponds to a low-level and local behavior that the units will follow.
Here is the list of available behaviors:
- attack_in_close_range: the unit attacks the enemies in close range if there are enemies or moves toward the target position if there are no enemies
- attack_static: the unit attacks the enemies without moving if there are enemies or moves toward the target position if there are no enemies
- attack_in_long_range: the unit attacks the enemies in long-range if there are enemies or moves toward the target position if there are no enemies

Apart from stand, all those behaviors will only be active if an enemy is in sight. Otherwise, they will move to a target position if you set one in the plan and stand if you do not set a target position.
Remember that units collide and push each other, so ignoring the enemies may not be the fastest way to reach a target position if there are enemies on the way who could block you.

The syntax format of a step is the following:
Step ID: (where you replace ID with the integer ID of the step)
prerequisites: [s_1, s_2, ..., s_n] (where the s_i correspond to the prerequisites steps IDs. Note that the list can be empty. In that case, simply write [])
objective: position [or] elimination [u_1, u_2, ..., u_n] (where the u_i correspond to the units integers IDs given in the game state description)
(At least one list of units and their assigned behavior and target position, but there can be as many groups as there are units, as one unit can belong to two groups:)
units: all (if all allies units are concerned) or [u_1, u_2, ..., u_n] (where the u_i correspond to the units integers IDs given in the game state description)
- target position: (x, y) (The integer x and y coordinate on the map.)
- behavior: behavior_name target_1 target_2 ... target_n (where behavior_name is an available behavior and target_1 to target_n are the targeted unit types or just the keyword "any" if the behavior targets any unit_type)

IMPORTANT:
- All positions (x, y) must be integers. If you want to give float positions, convert them first into integers.
- Do not add comments to the plan specification, as it interferes with the parser. If you want to give comments, give them outside the "BEGIN PLAN" and "END PLAN".
- A unit can only belong to one group of units for the same step

## Example of a valid detailed plan:

BEGIN PLAN
Step 0:
prerequisites: []
objective: position
units: all
- target position: (24, 14)
- behavior: attack_static any

Step 1:
prerequisites: [0]
objective: position
units: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- target position: (24, 16)
- behavior: attack_static any
units: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
- target position: (24, 14)
- behavior: attack_static any

Step 2:
prerequisites: [1]
objective: elimination all
units: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- target position: (24, 24)
- behavior: attack_in_close_range sniper soldier
units: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
- target position: (24, 24)
- behavior: attack_in_long_range soldier
END PLAN

START EACH LINE OF THE PLAN WITHOUT SPACES!

## List of planning mistakes you should avoid

For the position objective, you check that no enemies can block access to the target position (as the units can push and block each other).
Because, in that case, the units could never reach the destination.

A series of position objectives with different groups of units waiting for each other (through a chain of prerequisites) is suboptimal as all the different units can move simultaneously.
A simple improvement is to regroup all those positions in one step.

As the different steps have prerequisites, you can check the conditional relationship between them and check that all the units have an assigned behavior at every time step.
For example, each unit is in one of the unit groups for each step. If two (or more) steps can be active simultaneously (because they have the exact prerequisites), it is enough that each unit is at least in one of them.

For the elimination objective, ensure the target position is close enough to the targeted enemy units so that the units can effectively target them.
Also, ensure that at least one group of units exhibits offensive behavior; otherwise, the enemy units will never be eliminated.

As elimination objectives already have a default target position, it may be unnecessary to set a position objective before an elimination objective.
A simple improvement can be to remove the position objective and set the target position of the elimination objective to the same target position.
"""


################################################################################

model = "gpt-3.5-turbo"
config = {
    "model": model,
    "temperature": 0,
    "max_tokens": 4096,
    "instruction": instruction,
}


def current_state_to_txt(game):
    specific_env, state = game.env, game.state
    txt = ""
    txt += "Units:\n"
    for team, n, id_offset in zip(
        ["ally", "enemy"], [specific_env.num_allies, specific_env.num_enemies], [0, specific_env.num_allies]
    ):
        for unit_id in range(n):
            pos = state.unit_positions[unit_id + id_offset]
            health = int(state.unit_health[unit_id + id_offset])
            unit_type_idx = state.unit_types[unit_id + id_offset]
            max_health = int(specific_env.unit_type_health[unit_type_idx])
            if health == 0:
                txt += f"{team} {unit_id}: dead\n"
            else:
                pos_txt = f"position = ({pos[0]:2.1f}, {pos[1]:2.1f})"
                health_txt = f"health/max health) = {health}/{max_health}"
                type_txt = f"type = {specific_env.unit_type_names[unit_type_idx]}"
                txt += f"{team} {unit_id}: {type_txt}; {pos_txt}; {health_txt}\n"
    txt += map_description
    return [system_prompt(txt)]


map_description = f"""
## Pertinent elements on the map:
West forest: trees at (0, 0) - (20, 100)
East River: water at (80, 0) - (90, 100)
East River's Bridge: normal at (80, 10) - (90, 20)
"""
