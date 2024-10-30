# %% main.py
#   This file contains the FastAPI server that serves the Parabellum environment.
# by: Noah Syrkis

# %% Imports
import uuid
import asyncio
import time
import json
import llllll as ll
from dataclasses import field


from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from jax import random, jit, vmap, lax, tree_util
import jax.numpy as jnp
import parabellum as pb
from jax_tqdm import scan_tqdm
from functools import partial

# %% FastAPI server
app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)
games = {}
sleep_time = 0.1


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
    water = (1 * env.terrain.water).T
    trees = (2 * env.terrain.forest).T
    walls = (3 * env.terrain.building).T
    return {"unit_type_info": unit_type_info, "terrain": ((water + trees + walls).clip(0, 3)).tolist()}


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
    def __init__(
        self,
        id,
        env,
        step,
        state,
        obs,
        rng,
        env_info,
        agent_info,
        ally_plan,
        enemy_plan,
        bt_name_dict,
        bts_bank_variant_subsets,
    ):
        self.id = id
        self.env = env
        self.step_fn = step
        self.state: Optional[pb.State] = None
        self.rng = rng
        self.running = False
        self.terminal = False
        self.state_queue = asyncio.Queue()
        self.obs = obs
        self.env_info = env_info
        self.agent_info = agent_info
        self.ally_plan = ally_plan
        self.enemy_plan = enemy_plan
        self.direction_maps: Dict = {}
        self.assigned_bts = {a: 4 for a in env.agents}
        self.bt_name_dict = bt_name_dict
        self.bts_bank_variant_subsets = bts_bank_variant_subsets
        # self.plan = []

        # self.step_fn = jit(env.step)


# %% Routes
@app.post("/games/create/{place}")
async def game_create(place):
    game_id = str(uuid.uuid4())
    if game_id in games:
        return game_id
    # bts_txt = ["A (move north)", "A (move east)"]
    bts_fns, bts_bank_variant_subsets, bt_name_dict = ll.bts.bt_bank_jit()
    # bt_fns = ll.act.bts_fn(bts_bank)
    env, env_info, agent_info = ll.env.create_env(place)
    # assigned_bts = {a: int(a.startswith("e")) for a in env.agents}

    enemy_plan = ll.plan.parse_plan(
        ll.plan.enemy_plan,
        env.num_allies,
        env.num_enemies,
        3,
        ["position", "eliminate"],
        ll.bts.LLM_BTs,
        bts_bank_variant_subsets,
        for_ally=False,
    )

    ally_plan = ll.plan.parse_plan(
        ll.plan.ally_plan,
        env.num_allies,
        env.num_enemies,
        2,
        ["position", "eliminate"],
        ll.bts.LLM_BTs,
        bts_bank_variant_subsets,
    )

    step = ll.act.step_fn(env, env_info, agent_info, bts_fns)
    rng, key = random.split(random.PRNGKey(0))
    obs, state = env.reset(key)
    game = Game(
        game_id,
        env,
        step,
        state,
        obs,
        rng,
        env_info,
        agent_info,
        ally_plan,
        enemy_plan,
        bt_name_dict,
        bts_bank_variant_subsets,
    )
    ll.plan.initialize_plan(game)  # object hidden state stuff.
    games[game_id] = game
    return game_id, game_info_fn(env)


async def game_loop(game: Game, websocket: WebSocket):
    print(f"Starting game loop for game {game.id}")
    while game.running and not game.terminal:
        # new_state = step_game(game)
        ll.plan.apply_plan(game)
        game.rng, key = random.split(game.rng)
        new_obs, new_state, actions = game.step_fn(game.obs, game.state, key, game.assigned_bts)
        game.state = new_state
        game.terminal = new_state.terminal  # Convert to Python bool
        game.obs = new_obs

        state_dict = {
            "unit_positions": new_state.unit_positions.tolist(),
            "unit_alive": new_state.unit_alive.tolist(),
            "unit_teams": new_state.unit_teams.tolist(),
            "unit_health": new_state.unit_health.tolist(),
            "unit_types": new_state.unit_types.tolist(),
            "unit_weapon_cooldowns": new_state.unit_weapon_cooldowns.tolist(),
            "prev_attack_actions": new_state.prev_attack_actions.tolist(),
            "time": new_state.time.item(),  # type: ignore
            "terminal": new_state.terminal.item(),  # type: ignore
        }

        try:
            await websocket.send_json(state_dict)
            print(f"JSON message sent for game {game.id}")
        except WebSocketDisconnect:
            print(f"WebSocket for game {game.id} disconnected")
            break
        except Exception as e:
            print(f"Error in WebSocket for game {game.id}: {str(e)}")
            break

        await asyncio.sleep(sleep_time)


# def step_game(game: Game) -> pb.State:
#     game.rng, step_key = random.split(game.rng)
#     act_keys = random.split(step_key, len(game.env.agents))
#     actions = action_fn(game.env, game.state, act_keys)
#     obs, new_state, reward, done, infos = game.step_fn(step_key, game.state, actions)
#     print(f"Step completed for game {game.id}: time={new_state.time}")
#     return new_state


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
        "time": state.time,  # type: ignore
        "terminal": state.terminal,  # type: ignore
    }
    return {"obs": obs, "state": state_dict}


@app.websocket("/ws/{game_id}")
async def websocket_endpoint(websocket: WebSocket, game_id: str):
    await websocket.accept()
    print(f"WebSocket connected for game {game_id}")
    if game_id in games:
        game = games[game_id]
        game.running = True
        try:
            await game_loop(game, websocket)
        finally:
            game.running = False
            print(f"Game loop ended for game {game_id}")


@app.post("/games/{game_id}/start")
async def start_game(game_id: str):
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")
    game = games[game_id]
    game.running = True
    return {"message": "Game started"}


@app.post("/games/{game_id}/pause")
async def pause_game(game_id: str):
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")
    game = games[game_id]
    if game.running:
        game.running = False
        return {"message": "Game paused successfully"}
    else:
        return {"message": "Game is already paused"}


@app.post("/games/{game_id}/step")
async def step_game_endpoint(game_id: str):
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")
    game = games[game_id]
    if game.terminal or not game.state:
        raise HTTPException(status_code=400, detail="Game is already finished or not initialized")
    # Execute a single step
    ll.plan.apply_plan(game)
    game.rng, key = random.split(game.rng)
    new_obs, new_state, actions = game.step_fn(game.obs, game.state, key, game.assigned_bts)
    game.state = new_state
    game.terminal = new_state.terminal  # Convert to Python bool
    game.obs = new_obs
    state_dict = {
        "unit_positions": new_state.unit_positions.tolist(),
        "unit_alive": new_state.unit_alive.tolist(),
        "unit_teams": new_state.unit_teams.tolist(),
        "unit_health": new_state.unit_health.tolist(),
        "unit_types": new_state.unit_types.tolist(),
        "unit_weapon_cooldowns": new_state.unit_weapon_cooldowns.tolist(),
        "prev_attack_actions": new_state.prev_attack_actions.tolist(),
        "time": new_state.time.item(),  # type: ignore
        "terminal": new_state.terminal.item(),  # type: ignore
    }
    return JSONResponse(content={"state": state_dict}, status_code=200)


@app.post("/games/{game_id}/quit")
async def quit_game(game_id: str):
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")
    game = games.pop(game_id, None)
    if game:
        game.running = False
        print(f"Game {game_id} is terminated")
        return {"message": "Game terminated"}
    else:
        return {"message": "Game is not running or already terminated"}


#################### LLM API ####################
async def send_to_llm(message: str, game: Game) -> str:
    game_info = ll.llm.current_state_to_txt(game)  # [{"role": "system", "content": "You are a helpful assistant."}]
    return ll.llm.discuss_plan_with_LLM(ll.llm.config, message, [], game_info)["answer"]


class MessageRequest(BaseModel):
    message: str


@app.post("/games/{game_id}/process-message")  # Changed route to include game_id
async def process_message(game_id: str, request: MessageRequest):
    message = request.message
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")
    game = games[game_id]
    processed_message = await send_to_llm(message, game)
    _plan = ll.llm.handle_LLM_plan(game, processed_message)
    if _plan is not None:
        games[game_id].ally_plan = _plan
    return {"response": processed_message}


# def action_fn(env, bts, state, act_keys):
#     actions = {agent: env.action_space(agent).sample(step_key) for agent, step_key in zip(env.agents, act_keys)}
#     return actions
