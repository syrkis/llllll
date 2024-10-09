// store.ts
import { writable } from "svelte/store";
import type { State, Scenario } from "$lib/types";
import type { ScaleLinear } from "d3-scale";

export interface GameStore {
  gameId: string | null;
  currentState: State | null;
  gameInfo: Scenario | null;
}

const initialGameStore: GameStore = {
  gameId: null,
  currentState: null,
  gameInfo: null,
};

function createGameStore() {
  const { subscribe, set, update } = writable<GameStore>(initialGameStore);

  return {
    subscribe,
    setGame: (gameId: string, gameInfo: Scenario) => update((state) => ({ ...state, gameId, gameInfo })),
    setState: (currentState: State) => update((state) => ({ ...state, currentState })),
    reset: () => set(initialGameStore),
  };
}

export const gameStore = createGameStore();

export const scale = writable<ScaleLinear<number, number> | null>(null);
