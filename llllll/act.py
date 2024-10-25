# %% actions.py
#   llllll actions based on btc2sim


# %% Imports
import btc2sim
import jax
from jax import random, vmap, tree_util
from functools import partial
import numpy as np
import jax.numpy as jnp
from chex import dataclass
from dataclasses import field
from llllll import bts
# import llllll as ll


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
