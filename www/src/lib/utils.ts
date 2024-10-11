import { viewportHeight, gameStore, scale } from "$lib/store";
import { get } from "svelte/store";
import * as d3 from "d3";
import { updateVisualization } from "$lib/plots";

let resizeTimeout: ReturnType<typeof setTimeout>;

export function handleResize() {
  if (typeof window !== "undefined") {
    clearTimeout(resizeTimeout);

    resizeTimeout = setTimeout(() => {
      const newVh = Math.max(document.documentElement.clientHeight || 0, window.innerHeight || 0) * 0.96;
      viewportHeight.set(newVh);

      const { gameInfo } = get(gameStore);
      console.log("Current GameInfo:", gameInfo);
      if (!gameInfo || !gameInfo.terrain) {
        console.warn("GameInfo or terrain data is missing");
        return;
      }

      const gridSize = gameInfo.terrain.length;
      const newScale = d3.scaleLinear().domain([0, gridSize]).range([0, newVh]);

      const currentScale = get(scale);
      // Compare domain and range instead of the scale itself for accuracy
      if (!currentScale || !areScalesEqual(currentScale, newScale)) {
        scale.set(newScale);
        updateVisualization();
      }
    }, 200); // Debounce time is set to 200ms
  }
}

// Helper function to compare scales
function areScalesEqual(scale1: d3.ScaleLinear<number, number>, scale2: d3.ScaleLinear<number, number>): boolean {
  return (
    scale1.domain()[0] === scale2.domain()[0] &&
    scale1.domain()[1] === scale2.domain()[1] &&
    scale1.range()[0] === scale2.range()[0] &&
    scale1.range()[1] === scale2.range()[1]
  );
}
