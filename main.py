# %% api.py
#   This file contains the FastAPI server that serves the Parabellum environment.
# by: Noah Syrkis

# Imports
import uuid
from collections import namedtuple
from functools import partial
from dataclasses import asdict

import cv2
import parabellum as pb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from jax import random, tree
from omegaconf import OmegaConf

# %% Types
Game = namedtuple("Game", ["env", "scene", "step_fn", "step_seq"])
Step = namedtuple("Step", ["rng", "obs", "state", "action"])


# Configure CORS
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# %% Globals
games = {}
sleep_time = 0.1
n_steps = 100


# %% Functions
def action_fn(env, rng):  # should be in btc2sim
    coord = random.normal(rng, (env.num_units, 2))
    shoot = random.bernoulli(rng, 0.5, shape=(env.num_units,))
    return pb.types.Action(coord=coord, shoot=shoot)


def step_fn(env, scene, state, rng):  # should also be in btc2sim
    action = action_fn(env, rng)
    obs, state = env.step(rng, scene, state, action)
    return action, obs, state


# %% End points
@app.get("/init/{place}")
def init(place: str):  # should inlcude settings from frontend
    game_id = str(uuid.uuid4())
    cfg = OmegaConf.load("conf.yaml")
    env, scene = pb.env.Env(cfg=cfg), pb.env.scene_fn(cfg)
    step = partial(step_fn, env, scene)
    games[game_id] = Game(env, scene, step, [])  # <- state_seq list
    terrain = cv2.resize(np.array(scene.terrain.building), dsize=(100, 100)).tolist()
    return {"game_id": game_id, "terrain": terrain, "size": cfg.size, "teams": scene.unit_teams.tolist()}


@app.get("/reset/{game_id}")
def reset(game_id: str):
    rng, key = random.split(random.PRNGKey(0))
    obs, state = games[game_id].env.reset(rng=key, scene=games[game_id].scene)
    games[game_id].step_seq.append(Step(rng, obs, state, None))
    return {"state": asdict(tree.map(lambda x: x.tolist(), state)) | {"step": len(games[game_id].step_seq)}}


@app.get("/step/{game_id}")
def step(game_id: str):
    rng, key = random.split(games[game_id].step_seq[-1].rng)
    action, obs, state = games[game_id].step_fn(games[game_id].step_seq[-1].state, key)
    games[game_id].step_seq.append(Step(rng, obs, state, action))
    return {"state": asdict(tree.map(lambda x: x.tolist(), state)) | {"step": len(games[game_id].step_seq)}}


@app.post("/close/{game_id}")
async def close(game_id: str):
    del games[game_id]


## piece api
## chat api
