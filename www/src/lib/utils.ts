// src/lib/utils.ts
import { viewportHeight, gameStore, scale } from "$lib/store";
import { get } from "svelte/store";
import * as d3 from "d3";
import { updateVisualization } from "$lib/plots";

export function handleResize() {
  if (typeof window !== "undefined") {
    const newVh = Math.max(document.documentElement.clientHeight || 0, window.innerHeight || 0) * 0.96;
    viewportHeight.set(newVh);

    const { gameInfo } = get(gameStore);
    console.log("Current GameInfo:", gameInfo);
    if (!gameInfo || !gameInfo.terrain || !gameInfo.terrain.walls) {
      console.warn("GameInfo or terrain data is missing");
      return;
    }

    // Assuming terrain.walls contains the grid size information
    const gridSize = gameInfo.terrain.walls.length > 0 ? gameInfo.terrain.walls[0].length : 100;

    scale.set(d3.scaleLinear().domain([0, gridSize]).range([0, newVh]));
    updateVisualization();
  }
}
