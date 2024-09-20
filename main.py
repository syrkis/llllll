# main.py
#    FastAPI app
# by: Noah Syrkis


# Imports ######################################################################
from fastapi import FastAPI, Cookie, Body
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from geopandas.geoseries import Dict
from pydantic import BaseModel
from typing import Union
import gymnasium as gym
from jax import Array, random, tree_util
import jaxmarl
from jaxmarl.environments import smax
import parabellum as pb
from chex import dataclass
import uuid
from typing import Dict, List
import llllll


# FastAPI app ##################################################################
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Svelte dev server port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Game sessions ################################################################
game_sessions = {}


class GameStartRequest(BaseModel):
    scenario: str


# Functions ####################################################################
def get_game_session(idx: str):
    if idx not in game_sessions:
        raise HTTPException(status_code=404, detail="Game session not found")
    return game_sessions[idx]


def create_cookie(response: JSONResponse, session_id: str):
    response.set_cookie(key="session_id", value=session_id, httponly=True)
    return response


def jax_to_json(jax_data):
    return tree_util.tree_map(lambda x: x.tolist(), jax_data)


# Routes #######################################################################
@app.post("/assign_cookie")
def assign_cookie():
    session_id = str(uuid.uuid4())
    response = JSONResponse(content={"session_id": session_id})
    return create_cookie(response, session_id)


@app.post("/start_game")
async def start_game(request: GameStartRequest, session_id: str = Cookie(None)):
    if session_id is None:
        raise HTTPException(status_code=401, detail="No session_id provided")

    # Add "TEST" to the game_sessions dictionary with the cookie as the key

    scenario = smax.map_name_to_scenario(request.scenario)
    rng, key = random.split(random.PRNGKey(0))
    env = jaxmarl.make("SMAX", scenario=scenario)
    obs, state = env.reset(key)
    game_state = dict(
        obs=jax_to_json(obs), state=jax_to_json(state), rng=random.PRNGKey(0)
    )
    game_session = {"game_state": game_state, "env": env}
    game_sessions[session_id] = game_session
    return game_sessions[session_id]["game_state"]
