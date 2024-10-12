// types.ts
import type * as d3 from "d3";

export interface Scenario {
  unit_type_info: {
    unit_type_attack_ranges: number[];
    unit_type_sight_ranges: number[];
    unit_type_radiuses: number[];
    unit_type_health: number[];
  };
  terrain: number[][]; // Update to use a unified terrain matrix
}

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

// Removed the old separate terrain types
// Use a single terrain matrix instead
/* export type GridData = {
  water: boolean[][];
  walls: boolean[][];
  trees: boolean[][];
}; */

export type GridData = number[][]; // Unified grid data for terrain types

export type SVGSelection = d3.Selection<SVGSVGElement, unknown, null, undefined>;

export type CellVisualizationConfig = {
  size: number;
  offset: number;
  shape: "rect" | "circle";
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
