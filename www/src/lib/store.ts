// In store.ts
import { writable } from "svelte/store";
import type { State, Scenario, GridData } from "$lib/types";
import type { ScaleLinear } from "d3-scale";
import { emptyState } from "./types"; // import emptyState from types.ts

export interface GameStore {
  gameId: string | null;
  currentState: State | null;
  gameInfo: Scenario | null;
  terrain: GridData | null;
}

const initialTerrain: number[][] = Array.from({ length: 100 }, (_, rowIndex) =>
  Array.from({ length: 100 }, (_, colIndex) => {
    return rowIndex % 3 === 0 && colIndex % 3 === 0 ? 2 : 0;
  }),
);

const initialGameStore: GameStore = {
  gameId: null,
  currentState: emptyState,
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
    reset: () => set({ ...initialGameStore, currentState: emptyState }),
  };
}

export const gameStore = createGameStore();

export const scale = writable<ScaleLinear<number, number> | null>(null);

interface ChessPiece {
  name: string;
  symbol: string;
  x: number;
  y: number;
  active: boolean;
}

const initialPieces: ChessPiece[] = [
  { name: "King", symbol: "♔", x: 0, y: 0, active: false },
  { name: "Queen", symbol: "♕", x: 0, y: 0, active: false },
  { name: "Rook", symbol: "♖", x: 0, y: 0, active: false },
  { name: "Bishop", symbol: "♗", x: 0, y: 0, active: false },
  { name: "Knight", symbol: "♘", x: 0, y: 0, active: false },
  { name: "Pawn", symbol: "♙", x: 0, y: 0, active: false },
];

export const piecesStore = writable<ChessPiece[]>(initialPieces);

export interface TransitionDurations {
  shape: number;
  streak: number;
  terrain: number;
  marker: number;
}

const defaultTransitionDurations: TransitionDurations = {
  shape: 500, // For unit shapes
  streak: 500, // For attack streaks
  terrain: 1000, // For terrain updates
  marker: 500, // For markers/pieces
};

export const transitionDurations = writable<TransitionDurations>(defaultTransitionDurations);
