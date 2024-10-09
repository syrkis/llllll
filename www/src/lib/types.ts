import type * as d3 from "d3";

export interface EnvInfo {
  num_allies: number;
  num_enemies: number;
  unit_type_attack_ranges: number[];
  unit_type_sight_ranges: number[];
  unit_type_radiuses: number[];
  unit_type_health: number[];
}

export interface UnitData {
  position: number[];
  team: number;
  type: number;
  health: number;
  attack: number;
}

export interface State {
  unit_positions: number[][][];
  unit_alive: number[][];
  unit_teams: number[][];
  unit_health: number[][];
  unit_types: number[][];
  unit_weapon_cooldowns: number[][];
  prev_movement_actions: number[][][];
  prev_attack_actions: number[][];
  time: number[];
  terminal: number[];
}

export type SVGSelection = d3.Selection<SVGSVGElement, unknown, null, undefined>;

export type GridData = {
  solid: boolean[][];
  water: boolean[][];
  trees: boolean[][];
};

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
