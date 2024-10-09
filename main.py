# main.py
#   This file contains the FastAPI server that serves the Parabellum environment.
# by: Noah Syrkis

# %% Imports
import uuid
import asyncio

# WebSocketDisconnect
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from jax import random, jit, vmap, lax, tree_util
import jax.numpy as jnp
import parabellum as pb
from jax_tqdm import scan_tqdm
from functools import partial

# %% FastAPI server
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
games = {}


# %% Functions
def step(carry, xs, env):
    (obs, state), (time_step, rng) = carry, xs  # first element is the time step
    step_key, *act_keys = random.split(rng, 1 + len(env.agents))
    actions = {a: env.action_space(a).sample(act_keys[i]) for i, a in enumerate(env.agents)}
    new_obs, new_state, reward, done, infos = env.step(step_key, state, actions)
    return (new_obs, new_state), (step_key, state, actions)


def trajectory_fn(rng, env, state):  # for running one game quickly
    rng, key = random.split(rng)
    obs, state = env.reset(key)
    xs = (jnp.arange(100), random.split(rng, 100))
    (obs, state), state_seq = lax.scan(partial(step, env), (obs, state), xs)
    return state_seq


def game_info_fn(env):
    unit_type_info = tree_util.tree_map(
        lambda x: x.tolist(),
        {
            "unit_type_attack_ranges": env.unit_type_attack_ranges,
            "unit_type_sight_ranges": env.unit_type_sight_ranges,
            "unit_type_radiuses": env.unit_type_radiuses,
            "unit_type_health": env.unit_type_health,
        },
    )
    terrain = {
        "water": env.terrain.water.T.tolist(),
        "walls": env.terrain.building.T.tolist(),
        "trees": env.terrain.forest.T.tolist(),
    }
    return {"unit_type_info": unit_type_info, "terrain": terrain}


# %% Classes
class Action(BaseModel):
    action: int


class State(BaseModel):
    unit_positions: List[List[float]]
    unit_types: List[int]
    unit_health: List[float]


class GameState(BaseModel):  # for the client
    actions: Dict[str, int]
    state: State
    # observation: list
    # reward: float
    # terminated: bool
    # truncated: bool
    # info: dict


# %% Routes
@app.post("/games/create/{place}")
async def game_create(place):
    game_id = str(uuid.uuid4())
    if game_id in games:  # we already have a game with this id
        return game_id
    rng = random.PRNGKey(0)
    scenario = pb.env.scenario_fn(place, 100)
    env = pb.Environment(scenario=scenario)
    obs, state = env.reset(rng)
    games[game_id] = {"env": env, "rng": rng, "current_state": state, "running": False, "terminal": False}
    print(f"Game {game_id} created")
    return game_id, game_info_fn(env)


@app.post("/games/{game_id}/start")
async def start_game(game_id: str):
    print("starting game")
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")
    games[game_id]["running"] = True
    asyncio.create_task(game_loop(game_id))
    return {"message": "Game started"}


async def game_loop(game_id: str):
    while games[game_id]["running"] and not games[game_id]["terminal"]:
        print("stepping game")
        state = step_game(game_id)
        games[game_id]["current_state"] = state
        await asyncio.sleep(1)  # type: ignore[awaitable-is-generator]


def step_game(game_id: str) -> pb.State:  # Assuming pb.State is the correct return type
    env = games[game_id]["env"]
    state = games[game_id]["current_state"]
    rng, step_key = random.split(games[game_id]["rng"])
    games[game_id]["rng"] = rng

    actions = {a: env.action_space(a).sample(random.split(step_key, 1)[0]) for a in env.agents}
    obs, new_state, reward, done, infos = env.step(step_key, state, actions)

    games[game_id]["terminal"] = done

    return new_state


@app.websocket("/ws/{game_id}")
async def websocket_endpoint(websocket: WebSocket, game_id: str):
    await websocket.accept()  # type: ignore[awaitable-is-generator]
    try:
        while True:
            if game_id in games and "current_state" in games[game_id]:
                state = games[game_id]["current_state"]
                state_dict = {
                    "unit_positions": state.unit_positions.tolist(),
                    "unit_alive": state.unit_alive.tolist(),
                    "unit_teams": state.unit_teams.tolist(),
                    "unit_health": state.unit_health.tolist(),
                    "unit_types": state.unit_types.tolist(),
                    "unit_weapon_cooldowns": state.unit_weapon_cooldowns.tolist(),
                    "prev_attack_actions": state.prev_attack_actions.tolist(),
                    "time": state.time,
                    "terminal": state.terminal,
                }
                await websocket.send_json(state_dict)  # type: ignore
            await asyncio.sleep(0.1)  # type: ignore
    except WebSocketDisconnect:
        print(f"WebSocket for game {game_id} disconnected")


@app.post("/games/{game_id}/reset")
async def game_reset(game_id: str):
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")
    env = games[game_id]["env"]
    rng, key = random.split(random.PRNGKey(0))
    obs, state = env.reset(key)
    games[game_id]["rng"] = rng
    games[game_id]["current_state"] = state
    games[game_id]["running"] = False
    games[game_id]["terminal"] = False

    obs = tree_util.tree_map(lambda x: x.tolist(), obs)
    state_dict = {
        "unit_positions": state.unit_positions.tolist(),
        "unit_alive": state.unit_alive.tolist(),
        "unit_teams": state.unit_teams.tolist(),
        "unit_health": state.unit_health.tolist(),
        "unit_types": state.unit_types.tolist(),
        "unit_weapon_cooldowns": state.unit_weapon_cooldowns.tolist(),
        "prev_attack_actions": state.prev_attack_actions.tolist(),
        "time": state.time,
        "terminal": state.terminal,
    }
    return {"obs": obs, "state": state_dict}


# @app.post("/games/{game_id}/step")
# async def game_step(game_id: str, action: Action):
#     env = games[game_id]["env"]
#     obs, state = games[game_id]["obss"][-1], games[game_id]["states"][-1]
#     rng, key = random.split(games[game_id]["game_state_rng"][-1])
#     action_keys = random.split(key, len(env.agents))
#     actions = {a: env.action_space(a).sample(random.split(action_keys[i], 1)[0]) for i, a in enumerate(env.agents)}
#     new_obs, new_state, reward, done, infos = env.step(key, state, actions)
#     games[game_id]["states"].append(new_state)
#     games[game_id]["obss"].append(new_obs)
#     games[game_id]["rngs"].append(rng)
#     games[game_id]["rewards"].append(reward)
#     games[game_id]["terminated"].append(done)
#     games[game_id]["truncated"].append(False)
#     games[game_id]["infos"].append(infos)
#     return


@app.get("/games/{game_id}/state")
async def game_state(game_id: str):
    return games[game_id]["game_state"]


@app.delete("/games/{game_id}")
def game_delete(game_id: str):
    del games[game_id]


@app.get("/games")
def games_list():
    return list(games.keys())
