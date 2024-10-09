// src/lib/utils.ts
import { viewportHeight, gameInfo, scale } from "$lib/store"; // Import gameInfo instead of gridData
import { get } from "svelte/store";
import * as d3 from "d3";
import { updateVisualization } from "$lib/plots";

export function handleResize() {
  if (typeof window !== "undefined") {
    const newVh = Math.max(document.documentElement.clientHeight || 0, window.innerHeight || 0) * 0.96;
    viewportHeight.set(newVh); // Update the vh store

    const currentGameInfo = get(gameInfo);
    console.log("Current GameInfo:", currentGameInfo); // Add detailed logging
    if (!currentGameInfo || !currentGameInfo.terrain || !currentGameInfo.terrain.solid) {
      console.warn("GameInfo or terrain data is missing");
      return;
    }

    // Assuming terrain.solid contains the grid size information
    const gridSize = currentGameInfo.terrain.solid.length > 0 ? currentGameInfo.terrain.solid[0].length : 100; // Default to 100 if no data

    scale.set(d3.scaleLinear().domain([0, gridSize]).range([0, newVh]));
    updateVisualization(0);
  }
}
