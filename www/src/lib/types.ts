// types.ts
import type * as d3 from "d3";

// Add this to your types.ts file
// types.ts
export enum TerrainType {
  Empty = "empty",
  Water = "water",
  Trees = "trees",
  Walls = "walls",
  Solid = "solid",
}

// Add this function to convert from API numbers to TerrainType
export function numberToTerrainType(value: number): TerrainType {
  switch (value) {
    case 0:
      return TerrainType.Empty;
    case 1:
      return TerrainType.Water;
    case 2:
      return TerrainType.Trees;
    case 3:
      return TerrainType.Walls;
    default:
      return TerrainType.Empty;
  }
}

// Update the Scenario interface
export interface Scenario {
  unit_type_info: {
    unit_type_attack_ranges: number[];
    unit_type_sight_ranges: number[];
    unit_type_radiuses: number[];
    unit_type_health: number[];
  };
  terrain: TerrainType[][]; // Now uses TerrainType instead of number[][]
}

// Update GridData type
export type GridData = TerrainType[][];

export const emptyState: State = {
  unit_positions: [],
  unit_alive: [],
  unit_teams: [],
  unit_health: [],
  unit_types: [],
  unit_weapon_cooldowns: [],
  prev_movement_actions: [],
  prev_attack_actions: [],
  time: 0,
  terminal: true,
};

export interface Observation {
  [key: string]: number[] | number[][] | Observation;
}

export interface State {
  unit_positions: number[][];
  unit_alive: number[];
  unit_teams: number[];
  unit_health: number[];
  unit_types: number[];
  unit_weapon_cooldowns: number[];
  prev_movement_actions: number[][];
  prev_attack_actions: number[];
  time: number;
  terminal: boolean;
}

export interface UnitData {
  position: number[];
  team: number;
  type: number;
  health: number;
  maxHealth: number;
  attack: number;
}

export type SVGSelection = d3.Selection<SVGSVGElement, unknown, null, undefined>;

export type CellVisualizationConfig = {
  size: number;
  offset: number;
  shape: "circle" | "rect" | "triangle";
  className: string;
};

export interface CellConfig {
  size: number;
  offset: number;
  shape: string;
  className: string;
}

export interface SimulationConfig {
  cellSize: number;
  cellConfigs: {
    solid: CellConfig;
    water: CellConfig;
    tree: CellConfig;
  };
  speed: number; // Add speed property
}

export interface GameState {
  gameId: string | null;
  states: State | null;
  gameInfo: Scenario | null;
  currentStep: number;
  intervalId: ReturnType<typeof setInterval> | null;
  scale: d3.ScaleLinear<number, number> | null;
}

export interface ChessPiece {
  name: string;
  symbol: string;
  x: number;
  y: number;
  active: boolean;
}
