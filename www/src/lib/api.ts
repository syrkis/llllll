// api.ts
import type { State, Scenario, Observation } from "$lib/types";

export async function createGame(place: string): Promise<{ gameId: string; info: Scenario }> {
  const response = await fetch(`http://localhost:8000/games/create/${encodeURIComponent(place)}`, {
    method: "POST",
  });
  if (!response.ok) {
    throw new Error(`Failed to create game: ${response.statusText}`);
  }
  const [gameId, info] = await response.json();
  return { gameId, info };
}

export async function resetGame(gameId: string): Promise<{ obs: Observation; state: State }> {
  const response = await fetch(`http://localhost:8000/games/${gameId}/reset`, {
    method: "POST",
  });
  if (!response.ok) {
    throw new Error(`Failed to reset game: ${response.statusText}`);
  }
  return await response.json();
}

export async function startGame(gameId: string): Promise<void> {
  const response = await fetch(`http://localhost:8000/games/${gameId}/start`, {
    method: "POST",
  });
  if (!response.ok) {
    throw new Error(`Failed to start game: ${response.statusText}`);
  }
  const result = await response.json();
  if (result.error) {
    throw new Error(`Failed to start game: ${result.error}`);
  }
}

export async function pauseGame(gameId: string): Promise<void> {
  const response = await fetch(`http://localhost:8000/games/${gameId}/pause`, {
    method: "POST",
  });
  if (!response.ok) {
    throw new Error(`Failed to pause game: ${response.statusText}`);
  }
  const result = await response.json();
  if (result.error) {
    throw new Error(`Failed to pause game: ${result.error}`);
  }
}
