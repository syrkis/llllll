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
        groups = {}
        for i, bt_idx in enumerate(bt_idxs.values()):
            if bt_idx not in groups:
                groups[bt_idx] = []
            groups[bt_idx].append(i)
        actions = {}
        for grp_key, grp_indices in groups.items():
            grp_rng = agents_rng[jnp.array(grp_indices)]

            grp_obs = jnp.empty((len(grp_indices), env.obs_size))
            for i, idx in enumerate(grp_indices):
                grp_obs = grp_obs.at[i].set(obs[env.agents[idx]])

            grp_actions = vmap(bt_fns[grp_key], in_axes=(0, None, 0, 0))(
                grp_obs, env_info, split_agent_info(agents_info, grp_indices), grp_rng
            )

            for i, idx in enumerate(grp_indices):
                actions[env.agents[idx]] = grp_actions[i]
        obs, state, rewards, dones, infos = env.step(step_rng, state, actions)
        return obs, state, actions

    return step


def split_agent_info(agent_info, indices):
    indices = jnp.array(indices)
    new_agent_info = btc2sim.classes.AgentInfo(
        agent_id=agent_info.agent_id[indices],
        velocity=agent_info.velocity[indices],
        sight_range=agent_info.sight_range[indices],
        attack_range=agent_info.attack_range[indices],
        is_ally=agent_info.is_ally[indices],
        direction_map=agent_info.direction_map[indices],
    )
    return new_agent_info
