# main.py
#   This file contains the FastAPI server that serves the Parabellum environment.
# by: Noah Syrkis

# %% Imports
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from jax import random, jit, vmap, lax, tree_util
import jax.numpy as jnp
import parabellum as pb
from jax_tqdm import scan_tqdm
from functools import partial

# %% FastAPI server
app = FastAPI()
kwargs = dict(allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(CORSMiddleware, **kwargs)  # type: ignore
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
    games[game_id] = {"env": env}
    print(f"Game {game_id} created")
    # this should also return that things that don't change (env info)
    return game_id, game_info_fn(env)


# @app.post("/games/{game_id}")
# async def game(game_id: str):
#     if game_id not in games:
#         raise HTTPException(status_code=404, detail="Game not found")
#     if "game_state" not in games[game_id]:
#         raise HTTPException(status_code=404, detail="Game state not found")
#     return games[game_id]


@app.post("/games/{game_id}/reset")
async def game_reset(game_id: str):
    # strip game_id of quotes
    env = games[game_id]["env"]
    rng, key = random.split(random.PRNGKey(0))  # have this deppend on the game_id
    obs, state = env.reset(key)
    games[game_id]["states"] = [state]
    games[game_id]["obss"] = [obs]
    games[game_id]["rngs"] = [rng]
    games[game_id]["rewards"] = []
    games[game_id]["terminated"] = []
    games[game_id]["truncated"] = []
    games[game_id]["infos"] = []
    obs = tree_util.tree_map(lambda x: x.tolist(), obs)
    state = {
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
    return {"obs": obs, "state": state}


@app.post("/games/{game_id}/step")
async def game_step(game_id: str, action: Action):
    env = games[game_id]["env"]
    obs, state = games[game_id]["obss"][-1], games[game_id]["states"][-1]
    rng, key = random.split(games[game_id]["game_state_rng"][-1])
    actions = {a: action.action for a in env.agents}  # should the action be decided in the client? No
    new_obs, new_state, reward, done, infos = env.step(key, state, actions)
    games[game_id]["states"].append(new_state)
    games[game_id]["obss"].append(new_obs)
    games[game_id]["rngs"].append(rng)
    games[game_id]["rewards"].append(reward)
    games[game_id]["terminated"].append(done)
    games[game_id]["truncated"].append(False)
    games[game_id]["infos"].append(infos)
    return


@app.get("/games/{game_id}/state")
async def game_state(game_id: str):
    return games[game_id]["game_state"]


@app.delete("/games/{game_id}")
def game_delete(game_id: str):
    del games[game_id]


@app.get("/games")
def games_list():
    return list(games.keys())
