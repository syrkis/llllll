import { writable } from "svelte/store";
import type { State, Scenario, GridData } from "$lib/types";
import type { ScaleLinear } from "d3-scale";

export interface GameStore {
  gameId: string | null;
  currentState: State | null;
  gameInfo: Scenario | null;
  terrain: GridData | null; // Store terrain data
}

const initialTerrain: number[][] = Array.from({ length: 100 }, (_, rowIndex) =>
  Array.from({ length: 100 }, (_, colIndex) => {
    // Every 4th cell (in both row and column) will be 3, others 0
    return rowIndex % 4 === 0 && colIndex % 5 === 0 ? 3 : 0;
  }),
);

const initialGameStore: GameStore = {
  gameId: null,
  currentState: null,
  gameInfo: null,
  terrain: initialTerrain,
};

export const viewportHeight = writable<number>(0);

function createGameStore() {
  const { subscribe, set, update } = writable<GameStore>(initialGameStore);

  return {
    subscribe,
    setGame: (gameId: string, gameInfo: Scenario) => update((state) => ({ ...state, gameId, gameInfo })),
    setTerrain: (terrain: GridData) => update((state) => ({ ...state, terrain })),
    setState: (currentState: State) => update((state) => ({ ...state, currentState })),
    reset: () => set(initialGameStore),
  };
}

export const gameStore = createGameStore();

export const scale = writable<ScaleLinear<number, number> | null>(null);
