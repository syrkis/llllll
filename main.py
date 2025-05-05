# %% api.py
#   This file contains the FastAPI server that serves the Parabellum environment.
# by: Noah Syrkis

# Imports
import uuid
from collections import namedtuple
from dataclasses import asdict
from functools import partial

import btc2sim as b2s
import cv2
from typing import Tuple
from fastapi import Body
import jax.numpy as jnp
import numpy as np
import parabellum as pb
from einops import repeat
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from jax import random, tree, vmap
from omegaconf import OmegaConf

# %% Types
Game = namedtuple("Game", ["rng", "env", "scene", "step_fn", "gps", "step_seq"])
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

cfg = OmegaConf.load("conf.yaml")
env, scene = pb.env.Env(cfg=cfg), pb.env.scene_fn(cfg)
# scene.terrain.building = b2s.utils.scene_fn(scene.terrain.building)
rng, key = random.split(random.PRNGKey(0))
bt = tree.map(lambda x: repeat(x, f"... -> {cfg.num_units} ..."), b2s.dsl.txt2bts(open("bts.txt", "r").readline()))
i2p = sorted(["king", "queen", "rook", "bishop", "knight", "pawn"])
p2i = {p: i for i, p in enumerate(i2p)}
targets = jnp.int32(jnp.arange(6).repeat(env.num_units // 6)).flatten()


# %% Functions
def action_fn(env, obs, rng, gps, targets):  # should be in btc2sim
    rngs = random.split(rng, cfg.num_units)
    aux = vmap(b2s.act.action_fn, in_axes=(0, 0, 0, None, None, None, 0))
    return tree.map(jnp.squeeze, aux(rngs, obs, bt, env, scene, gps, targets))  # get rid of squeeze.


def step_fn(env, scene, obs, state, gps, targets, rng):  # should also be in btc2sim
    action = action_fn(env, obs, rng, gps, targets)
    obs, state = env.step(rng, scene, state, action)
    return action, obs, state


# %% End points
@app.get("/init/{place}")
def init(place: str):  # should inlcude settings from frontend
    game_id = str(uuid.uuid4())
    step = partial(step_fn, env, scene)
    rng = random.PRNGKey(0)
    gps = tree.map(jnp.zeros_like, b2s.gps.gps_fn(scene, jnp.int32(jnp.zeros((6, 2)))))
    games[game_id] = Game([rng], env, scene, step, gps, [])  # <- state_seq list
    terrain = cv2.resize(np.array(scene.terrain.building), dsize=(100, 100)).tolist()
    teams = scene.unit_teams.tolist()
    marks = {k: v for k, v in zip(i2p, gps.marks.tolist())}
    return {"game_id": game_id, "terrain": terrain, "size": cfg.size, "teams": teams, "marks": marks}


@app.get("/reset/{game_id}")
def reset(game_id: str):
    rng, key = random.split(games[game_id].rng[-1])
    obs, state = games[game_id].env.reset(rng=key, scene=games[game_id].scene)
    games[game_id].step_seq.append(Step(rng, obs, state, None))
    games[game_id].rng.append(rng)
    return {"state": asdict(tree.map(lambda x: x.tolist(), state)) | {"step": len(games[game_id].step_seq)}}


@app.get("/step/{game_id}")
def step(game_id: str):
    rng, key = random.split(games[game_id].step_seq[-1].rng)
    obs, state = games[game_id].step_seq[-1].obs, games[game_id].step_seq[-1].state
    gps = games[game_id].gps
    action, obs, state = games[game_id].step_fn(obs, state, gps, targets, key)
    games[game_id].step_seq.append(Step(rng, obs, state, action))
    return {"state": asdict(tree.map(lambda x: x.tolist(), state)) | {"step": len(games[game_id].step_seq)}}


@app.post("/close/{game_id}")
async def close(game_id: str):
    del games[game_id]


@app.post("/marks/{game_id}")
async def marks(game_id: str, marks: list = Body(...)):
    gps = b2s.gps.gps_fn(scene, jnp.int32(jnp.array(marks))[:, ::-1])
    # struct = tree.structure(gps)
    # print(tree.structure(tree.transpose(struct, None, gps)))
    # gps = tree.map(lambda x: x * ~(gps.marks == 0).all(), gps)
    games[game_id] = games[game_id]._replace(gps=gps)
    return {"marks": {k: v.tolist() for k, v in zip(i2p, gps.marks)}}
