import type * as d3 from "d3";
import { easeCubicInOut } from "d3";

interface TerrainCell {
  value: number;
  x: number;
  y: number;
}

export function createBackgroundGrid(
  svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
  terrainMatrix: number[][] | null,
  scale: d3.ScaleLinear<number, number> | null,
  isResize = false, // Do not explicitly annotate; TypeScript infers this as boolean from the default value
) {
  if (!terrainMatrix || !scale) {
    console.warn("Terrain matrix or scale is undefined");
    return;
  }

  const cellSize = scale(1) - scale(0);
  const maxSize = cellSize * 1; // Maximum size of the square, slightly smaller than cell size
  const minSize = 0.01; // Minimum size of the square for visibility

  const maxSizeScaleFactor = maxSize / 20;

  const terrainData: TerrainCell[] = terrainMatrix.flat().map((value, index) => ({
    value,
    x: index % terrainMatrix[0].length,
    y: Math.floor(index / terrainMatrix[0].length),
  }));

  const cells = svg
    .selectAll<SVGRectElement, TerrainCell>(".terrain-cell")
    .data(terrainData, (d: TerrainCell) => `${d.x}-${d.y}`);

  // Enter selection
  const enter = cells
    .enter()
    .append("rect")
    .attr("class", "terrain-cell ink")
    .attr("x", (d) => scale(d.x) + cellSize / 2)
    .attr("y", (d) => scale(d.y) + cellSize / 2)
    .attr("width", 0)
    .attr("height", 0);

  // Merge enter and update selections
  const merged = enter.merge(cells);

  if (isResize) {
    // No transition for resize
    merged
      .attr("x", (d) => {
        const size = Math.max(minSize, d.value ** 2 * maxSizeScaleFactor);
        return scale(d.x) + (cellSize - size) / 2;
      })
      .attr("y", (d) => {
        const size = Math.max(minSize, d.value ** 2 * maxSizeScaleFactor);
        return scale(d.y) + (cellSize - size) / 2;
      })
      .attr("width", (d) => Math.max(minSize, d.value ** 2 * maxSizeScaleFactor))
      .attr("height", (d) => Math.max(minSize, d.value ** 2 * maxSizeScaleFactor));
  } else {
    // Apply transition for updates
    merged
      .transition()
      .duration(1000)
      .ease(easeCubicInOut)
      .attr("x", (d) => {
        const size = Math.max(minSize, d.value ** 2 * maxSizeScaleFactor);
        return scale(d.x) + (cellSize - size) / 2;
      })
      .attr("y", (d) => {
        const size = Math.max(minSize, d.value ** 2 * maxSizeScaleFactor);
        return scale(d.y) + (cellSize - size) / 2;
      })
      .attr("width", (d) => Math.max(minSize, d.value ** 2 * maxSizeScaleFactor))
      .attr("height", (d) => Math.max(minSize, d.value ** 2 * maxSizeScaleFactor));
  }

  // Exit selection
  cells.exit().remove();
}
