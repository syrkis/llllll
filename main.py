import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from jax import random, jit, vmap, lax, tree_util
import jax.numpy as jnp
import parabellum as pb
from jax_tqdm import scan_tqdm


# place = "Vesterbro, Copenhagen, Denmark"
# scenario = pb.env.scenario_fn(place)  # this should use a cache
# env = pb.Environment(scenario=scenario)

# init = jit(env.reset)
# step = jit(env.step)

# rng, init_key, step_key, act_rng = random.split(random.PRNGKey(0), 4)
# obs, state = env.reset(init_key)


def trajectory_fn(rng, env, state):
    @scan_tqdm(100)
    def step_fn(carry, xs):
        (obs, state), (_, rng) = carry, xs
        step_key, *act_keys = random.split(rng, 1 + len(env.agents))
        actions = {a: env.action_space(a).sample(act_keys[i]) for i, a in enumerate(env.agents)}
        new_obs, new_state, reward, done, infos = env.step(step_key, state, actions)
        return (new_obs, new_state), (step_key, state, actions)

    rng, key = random.split(rng)
    obs, state = env.reset(key)
    xs = (jnp.arange(100), random.split(rng, 100))
    (obs, state), state_seq = lax.scan(step_fn, (obs, state), xs)

    return state_seq


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Add your Svelte app's URL here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Store active games
games = {}


class Action(BaseModel):
    action: int


class GameState(BaseModel):
    observation: list
    reward: float
    terminated: bool
    truncated: bool
    info: dict


@app.post("/start")
async def start_game():
    game_id = str(uuid.uuid4())
    rng = random.PRNGKey(uuid.getnode())
    place = "Vesterbro, Copenhagen, Denmark"
    scenario = pb.env.scenario_fn(place)
    env = pb.Environment(scenario=scenario)
    obs, state = env.reset(rng)
    rngs, states, actions = trajectory_fn(rng, env, state)
    return {
        "game_id": game_id,
        "rngs": tree_util.tree_map(lambda x: x.tolist(), rngs),
        "states": tree_util.tree_map(lambda x: x.tolist(), states),
        "actions": tree_util.tree_map(lambda x: x.tolist(), actions),
    }


# @app.post("/step/{game_id}")
# async def step(game_id: str, action: Action):
#     if game_id not in games:
#         raise HTTPException(status_code=404, detail="Game not found")

#     env = games[game_id]["env"]
#     observation, reward, terminated, truncated, info = env.step(action.action)

#     games[game_id]["observation"] = observation.tolist()
#     games[game_id]["info"] = info

#     return GameState(
#         observation=observation.tolist(),
#         reward=float(reward),
#         terminated=terminated,
#         truncated=truncated,
#         info=info,
#     )


# @app.get("/game_state/{game_id}")
# async def get_game_state(game_id: str):
#     if game_id not in games:
#         raise HTTPException(status_code=404, detail="Game not found")

#     return GameState(
#         observation=games[game_id]["observation"],
#         reward=0.0,  # We don't store reward in the game state
#         terminated=False,  # We don't store terminated in the game state
#         truncated=False,  # We don't store truncated in the game state
#         info=games[game_id]["info"],
#     )


# @app.delete("/end_game/{game_id}")
# async def end_game(game_id: str):
#     if game_id not in games:
#         raise HTTPException(status_code=404, detail="Game not found")

#     games[game_id]["env"].close()
#     del games[game_id]
#     return {"message": "Game ended successfully"}
