import { writable } from "svelte/store";
import type { State, GridData, EnvInfo } from "$lib/types";
import type { ScaleLinear } from "d3-scale";

export const gameId = writable<string | null>(null);
export const viewportHeight = writable<number>(0);
export const states = writable<State | null>(null);
export const gridData = writable<GridData>({ solid: [], water: [], trees: [] });
export const envInfo = writable<EnvInfo | null>(null);
export const currentStep = writable(0);
export const intervalId = writable<ReturnType<typeof setInterval> | null>(null);
export const scale = writable<ScaleLinear<number, number> | null>(null);
