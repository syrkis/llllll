// Basic game types
export type UnitType = "unit" | "building" | "resource";
export type CellType = "grass" | "water" | "mountain";

// Game entities
export interface Unit {
  id: number;
  x: number;
  y: number;
  size: number;
  health: number;
  type: UnitType;
}

export interface Cell {
  x: number;
  y: number;
  type: CellType;
}

// Game state structures
export interface Scene {
  location: string;
  terrain: Cell[][];
}

export interface State {
  units: Unit[][];
  step: number;
  pos: number[];
}

// API-related types
export interface RawInfo {
  terrain: number[][];
  [key: string]: unknown;
}

export interface RawState {
  unit_positions: number[]; // Raw unit position data from API
  step: number;
  [key: string]: unknown;
}

export interface ApiResponse<T> {
  state?: RawState;
  [key: string]: unknown;
}
