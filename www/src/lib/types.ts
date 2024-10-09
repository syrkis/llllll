// types.ts
import type * as d3 from "d3";

export interface Scenario {
  unit_type_info: {
    unit_type_attack_ranges: number[];
    unit_type_sight_ranges: number[];
    unit_type_radiuses: number[];
    unit_type_health: number[];
  };
  terrain: GridData;
}

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

export type GridData = {
  water: boolean[][];
  walls: boolean[][];
  trees: boolean[][];
};

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
