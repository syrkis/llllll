import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import gymnasium as gym

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


@app.post("/start_game")
async def start_game():
    game_id = str(uuid.uuid4())
    env = gym.make("CartPole-v1")
    initial_observation, info = env.reset()
    games[game_id] = {
        "env": env,
        "observation": initial_observation.tolist(),
        "info": info,
    }
    return {
        "game_id": game_id,
        "initial_observation": initial_observation.tolist(),
        "info": info,
    }


@app.post("/step/{game_id}")
async def step(game_id: str, action: Action):
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")

    env = games[game_id]["env"]
    observation, reward, terminated, truncated, info = env.step(action.action)

    games[game_id]["observation"] = observation.tolist()
    games[game_id]["info"] = info

    return GameState(
        observation=observation.tolist(),
        reward=float(reward),
        terminated=terminated,
        truncated=truncated,
        info=info,
    )


@app.get("/game_state/{game_id}")
async def get_game_state(game_id: str):
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")

    return GameState(
        observation=games[game_id]["observation"],
        reward=0.0,  # We don't store reward in the game state
        terminated=False,  # We don't store terminated in the game state
        truncated=False,  # We don't store truncated in the game state
        info=games[game_id]["info"],
    )


@app.delete("/end_game/{game_id}")
async def end_game(game_id: str):
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")

    games[game_id]["env"].close()
    del games[game_id]
    return {"message": "Game ended successfully"}
