// api.ts
import type { State, Scenario, Observation, numberToTerrainType } from "$lib/types";

const API_BASE_URL = "http://localhost:8000";

// api.ts
import { TerrainType, numberToTerrainType } from "$lib/types";

export async function createGame(place: string): Promise<{ gameId: string; info: Scenario }> {
  const response = await fetch(`${API_BASE_URL}/games/create/${encodeURIComponent(place)}`, {
    method: "POST",
  });
  if (!response.ok) {
    throw new Error(`Failed to create game: ${response.statusText}`);
  }
  const [gameId, rawInfo] = await response.json();

  // Convert the numeric terrain to TerrainType
  const info: Scenario = {
    ...rawInfo,
    terrain: rawInfo.terrain.map((row: number[]) => row.map((cell: number) => numberToTerrainType(cell))),
  };

  return { gameId, info };
}

export async function resetGame(gameId: string): Promise<{ obs: Observation; state: State }> {
  const response = await fetch(`${API_BASE_URL}/games/${gameId}/reset`, {
    method: "POST",
  });
  if (!response.ok) {
    throw new Error(`Failed to reset game: ${response.statusText}`);
  }
  return await response.json();
}

export async function startGame(gameId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/games/${gameId}/start`, {
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
  const response = await fetch(`${API_BASE_URL}/games/${gameId}/pause`, {
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

export async function stepGame(gameId: string): Promise<State> {
  const response = await fetch(`${API_BASE_URL}/games/${gameId}/step`, {
    method: "POST",
  });
  if (!response.ok) {
    throw new Error(`Failed to step game: ${response.statusText}`);
  }
  const result = await response.json();
  return result.state;
}

export async function quitGame(gameId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/games/${gameId}/quit`, {
    method: "POST",
  });
  if (!response.ok) {
    throw new Error(`Failed to quit game: ${response.statusText}`);
  }
}

// In api.ts
export async function sendMessage(gameId: string, message: string): Promise<string> {
  const response = await fetch(`${API_BASE_URL}/games/${gameId}/process-message`, {
    // Updated URL
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      message: message,
    }), // Only send message in body
  });

  if (!response.ok) {
    throw new Error(`Failed to process message: ${response.statusText}`);
  }

  const result = await response.json();
  return result.response;
}
