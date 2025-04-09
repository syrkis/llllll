# %% api.py
#   This file contains the FastAPI server that serves the Parabellum environment.
# by: Noah Syrkis

# Imports
import uuid
from collections import namedtuple
from functools import partial

import parabellum as pb
from fastapi import FastAPI
from jax import random, tree
from omegaconf import OmegaConf

# %% Types
Game = namedtuple("Game", ["rng", "env", "scene", "step_fn", "step_seq"])
Step = namedtuple("Step", ["rng", "obs", "state", "action"])


# %% Globals
app = FastAPI()
games = {}
sleep_time = 0.1
n_steps = 100


# %% Functions
def action_fn(env, rng):  # should be in btc2sim
    coord = random.normal(rng, (env.num_units, 2))
    kinds = random.bernoulli(rng, 0.5, shape=(env.num_units,))
    return pb.types.Action(coord=coord, kinds=kinds)


def step_fn(env, scene, state, rng):  # should also be in btc2sim
    action = action_fn(env, rng)
    obs, state = env.step(rng, scene, state, action)
    return action, obs, state


# %% Routes
@app.get("/init")
def init():  # should inlcude settings from frontend
    game_id = str(uuid.uuid4())
    cfg = OmegaConf.load("conf.yaml")
    env, scene = pb.env.Env(cfg=cfg), pb.env.scene_fn(cfg)
    rng, step = random.PRNGKey(0), partial(step_fn, env, scene)
    games[game_id] = Game(rng, env, scene, step, [])  # <- state_seq list
    print(scene.terrain.building.shape)
    return {"game_id": game_id}  # "terrain": scene.terrain.building.tolist()}


@app.get("/reset/{game_id}")
def reset(game_id: str):
    rng, key = random.split(games[game_id].rng)
    obs, state = games[game_id].env.reset(rng=key, scene=games[game_id].scene)
    games[game_id].step_seq.append(Step(rng, obs, state, None))
    return {"state": tree.map(lambda x: x.tolist(), state)}


@app.get("/step/{game_id}")
def step(game_id: str):
    rng, key = random.split(games[game_id].rng)
    action, obs, state = games[game_id].step_fn(games[game_id].step_seq[-1].state, key)
    games[game_id].step_seq.append(Step(rng, obs, state, action))
    return {"state": tree.map(lambda x: x.tolist(), state)}


@app.post("/close/{game_id}")
async def close(game_id: str):
    del games[game_id]
