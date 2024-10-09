# main.py
#   This file contains the FastAPI server that serves the Parabellum environment.
# by: Noah Syrkis

# %% Imports
import uuid
import asyncio

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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
games = {}


# %% Functions
def step(carry, xs, env):
    (obs, state), (time_step, rng) = carry, xs  # first element is the time step
    step_key, *act_keys = random.split(rng, 1 + len(env.agents))
    actions = {a: 2 for a in env.agents}
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


class Game:
    def __init__(self, id, env):
        self.id = id
        self.env = env
        self.state = None
        self.running = False
        self.terminal = False
        self.state_updated = asyncio.Event()
        self.rng = random.PRNGKey(0)


# %% Routes
@app.post("/games/create/{place}")
async def game_create(place):
    game_id = str(uuid.uuid4())
    if game_id in games:  # we already have a game with this id
        return game_id
    scenario = pb.env.scenario_fn(place, 100)
    env = pb.Environment(scenario=scenario)
    game = Game(game_id, env)
    game.state = game.env.reset(game.rng)[1]  # Get initial state
    games[game_id] = game
    print(f"Game {game_id} created")
    return game_id, game_info_fn(env)


@app.post("/games/{game_id}/start")
async def start_game(game_id: str):
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")
    game = games[game_id]
    if not game.running:
        game.running = True
        asyncio.create_task(game_loop(game))
    return {"message": "Game started"}


async def game_loop(game: Game):
    while game.running and not game.terminal:
        new_state = step_game(game)
        game.state = new_state
        game.state_updated.set()  # notify
        await asyncio.sleep(1)


def step_game(game: Game) -> pb.State:
    game.rng, step_key = random.split(game.rng)
    actions = {a: 2 for a in game.env.agents}
    obs, new_state, reward, done, infos = game.env.step(step_key, game.state, actions)

    game.terminal = new_state.terminal.all().item()
    print(f"Step completed for game {game.id}: terminal={game.terminal}, time={new_state.time}")

    return new_state


@app.get("/games/{game_id}/status")
async def game_status(game_id: str):
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")
    game = games[game_id]
    return {
        "running": game.running,
        "terminal": game.terminal,
        "time": game.state.time if game.state else None,
    }


@app.websocket("/ws/{game_id}")
async def websocket_endpoint(websocket: WebSocket, game_id: str):
    await websocket.accept()
    if game_id not in games:
        await websocket.close(code=4000, reason="Game not found")
        return
    game = games[game_id]
    try:
        while True:
            await game.state_updated.wait()  # Wait for the state to be updated
            game.state_updated.clear()  # Reset the event
            state_dict = {
                "unit_positions": game.state.unit_positions.tolist(),
                "unit_alive": game.state.unit_alive.tolist(),
                "unit_teams": game.state.unit_teams.tolist(),
                "unit_health": game.state.unit_health.tolist(),
                "unit_types": game.state.unit_types.tolist(),
                "unit_weapon_cooldowns": game.state.unit_weapon_cooldowns.tolist(),
                "prev_attack_actions": game.state.prev_attack_actions.tolist(),
                "time": game.state.time,
                "terminal": game.state.terminal.tolist(),
            }
            await websocket.send_json(state_dict)
    except WebSocketDisconnect:
        print(f"WebSocket for game {game_id} disconnected")


@app.post("/games/{game_id}/reset")
async def game_reset(game_id: str):
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")
    game = games[game_id]
    game.rng, key = random.split(random.PRNGKey(0))
    obs, state = game.env.reset(key)
    game.state = state
    game.running = False
    game.terminal = False

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


@app.get("/games/{game_id}/state")
async def game_state(game_id: str):
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")
    game = games[game_id]
    return {
        "unit_positions": game.state.unit_positions.tolist(),
        "unit_alive": game.state.unit_alive.tolist(),
        "unit_teams": game.state.unit_teams.tolist(),
        "unit_health": game.state.unit_health.tolist(),
        "unit_types": game.state.unit_types.tolist(),
        "unit_weapon_cooldowns": game.state.unit_weapon_cooldowns.tolist(),
        "prev_attack_actions": game.state.prev_attack_actions.tolist(),
        "time": game.state.time,
        "terminal": game.state.terminal.tolist(),
    }


@app.delete("/games/{game_id}")
def game_delete(game_id: str):
    if game_id in games:
        del games[game_id]
    return {"message": f"Game {game_id} deleted"}


@app.get("/games")
def games_list():
    return list(games.keys())
