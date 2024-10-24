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
from llllll import bts
# import llllll as ll

# %% Constants
default_plan = f"""Step 0:
prerequisites: []
objective: position
units: all
- target position: {(5, 5)}
- behavior: Stand
"""


# %% Functions
def step_fn(env, env_info, agents_info, bt_fns):
    def step(obs, state, rng, bt_idxs):
        if "world_state" in obs:
            obs.pop("world_state")
        act_rng, step_rng = random.split(rng)
        agents_rng = random.split(act_rng, env.num_agents)
        agents_rng = {agent: agents_rng[i] for i, agent in enumerate(agents_info.keys())}
        acts = tree_util.tree_map(
            lambda bt_idx, agent_obs, agent_info, agent_rng: jax.lax.switch(
                bt_idx, bt_fns, agent_obs, env_info, agent_info, agent_rng
            ),
            bt_idxs,
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
