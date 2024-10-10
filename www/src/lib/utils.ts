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
      const currentScale = scale.get();
      const newScale = d3.scaleLinear().domain([0, gridSize]).range([0, newVh]);

      // Only update if scale has changed
      if (currentScale !== newScale) {
        scale.set(newScale);
        updateVisualization();
      }
    }, 200); // Debounce time is set to 200ms
  }
}
