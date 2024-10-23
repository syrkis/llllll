# %% actions.py
#   llllll actions based on btc2sim


# %% Imports
from btc2sim import btc2sim
import jax
from jax import random, vmap, tree_util
from functools import partial
import numpy as np
import jax.numpy as jnp
from chex import dataclass
from dataclasses import field
from llllll.env import Objective
import llllll as ll

# %% Constants
default_plan = f"""Step 0:
prerequisites: []
objective: position
units: all
- target position: {(5, 5)}
- behavior: stand
"""


# %% Functions
def step_fn(env, env_info, agents_info, bt_fns, assigned_bts):
    def step(obs, state, rng):
        if "world_state" in obs:
            obs.pop("world_state")
        act_rng, step_rng = random.split(rng)
        agents_rng = random.split(act_rng, env.num_agents)
        agents_rng = {agent: agents_rng[i] for i, agent in enumerate(agents_info.keys())}
        acts = tree_util.tree_map(
            lambda bt_idx, agent_obs, agent_info, agent_rng: jax.lax.switch(
                bt_idx, bt_fns, agent_obs, env_info, agent_info, agent_rng
            ),
            assigned_bts,
            obs,
            agents_info,
            agents_rng,
        )
        obs, state, rewards, dones, infos = env.step(step_rng, state, acts)
        return obs, state, acts

    return step


def eval_bt(bt, obs, env_info, agent_info, rng):  # take actions for all agents in parallel
    acts = tree_util.tree_map(lambda x, i: bt(x, env_info, i, rng)[1], obs, agent_info)
    return acts


def bts_fn(bt_strs):
    dsl_trees = [btc2sim.dsl.parse(btc2sim.dsl.read(bt_str)) for bt_str in bt_strs]
    bts = [btc2sim.bt.seed_fn(dsl_tree) for dsl_tree in dsl_trees]
    return [partial(eval_bt, bt) for bt in bts]


def bt_fn(bt_str):
    dsl_tree = btc2sim.dsl.parse(btc2sim.dsl.read(bt_str))
    bt = btc2sim.bt.seed_fn(dsl_tree)
    return lambda obs, env_info, agent_info, rng: bt(obs, env_info, agent_info, rng)[1]


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


# +
def initialize_plan(game):
    for step in game.plan.values():
        for target in step.target_pos.values():
            compute_direction_map(game, target)
    for step in game.enemy_plan.values():
        for target in step.target_pos.values():
            compute_direction_map(game, target)


def compute_direction_map(game, target):
    target = (int(target[0]), int(target[1]))
    mask = jnp.logical_or(game.env.terrain.building, game.env.terrain.water)
    if target not in game.direction_maps:
        game.direction_maps[target] = ll.env.compute_bfs(mask, target)


def reset_plan(game):
    initialize_plan(game)
    for step in game.plan.values():
        step.done = False
    for step in game.enemy_plan.values():
        step.done = False


# -

# ### Execution


def execute_plan(game):
    if len(game.plan) > 0:
        initialize_plan(game)


def get_current_steps(game):
    current_ally_steps = []
    for step in game.plan.values():
        if not step.done and step.check_prerequisites(game.plan) and not step.check_objective(game):
            current_ally_steps.append(step)
    current_enemy_steps = []
    for step in game.enemy_plan.values():
        if not step.done and step.check_prerequisites(game.enemy_plan) and not step.check_objective(game):
            current_enemy_steps.append(step)
    return current_ally_steps, current_enemy_steps


def apply_plan(game):  # changes bt and target of each agent.
    current_ally_steps, current_enemy_steps = get_current_steps(game)
    if current_ally_steps == []:
        return 1
    elif current_enemy_steps == []:
        return -1
    for unit_id in range(
        game.env.num_agents
    ):  # reset the direction map of each unit so that it stands if no direction map is given and it uses the follow_direction atomic
        game.agents_info[
            f"ally_{unit_id}" if unit_id < game.env.num_allies else f"enemy_{unit_id-game.env.num_allies}"
        ].direction_map = jnp.ones(game.env.terrain.building.shape, dtype=jnp.int32) * 4
    for current_step in current_ally_steps + current_enemy_steps:
        for unit_id in current_step.units:
            game.assigned_BT_key[unit_id] = current_step.assigned_bt[unit_id]
            if unit_id in current_step.target_pos:
                target = current_step.target_pos[unit_id]
                if target not in game.direction_maps:
                    compute_direction_map(game, target)
                game.agents_info[
                    f"ally_{unit_id}" if unit_id < game.env.num_allies else f"enemy_{unit_id-game.env.num_allies}"
                ].direction_map = game.direction_maps[target]
    return 0


# ### Ask LLM


bts_bank = {}
bts_bank_variant_subsets = {}
for key, bt in ll.bts.handcrafted_bts.items():
    subsets, variants = ll.bts.compute_all_variants(bt["bt"], ll.bts.unit_types)
    bts_bank_variant_subsets[key] = subsets
    for i, bt_txt in enumerate(variants):
        name = key + (("_" + str(i)) if len(variants) > 1 else "")
        bts_bank[name] = bt_fn(bt_txt)


# +
true_enemies_subset = {"soldier", "sniper", "swat"}
unit_types_set = {"soldier", "sniper", "swat", "turret", "civilian"}


def find_bt_key(bt_name, subset):
    if len(bts_bank_variant_subsets[bt_name]) == 1:
        return bt_name
    else:
        return bt_name + "_" + str(bts_bank_variant_subsets[bt_name].index(subset))
