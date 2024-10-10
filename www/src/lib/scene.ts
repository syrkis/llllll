import type * as d3 from "d3";
import type { SVGSelection } from "$lib/types";

interface TerrainCell {
  value: number;
  x: number;
  y: number;
}

export function createBackgroundGrid(
  svg: d3.Selection<SVGSVGElement, unknown, HTMLElement, unknown>,
  terrainMatrix: number[][] | null,
  scale: d3.ScaleLinear<number, number> | null,
) {
  if (!terrainMatrix || !scale) {
    console.warn("Terrain matrix or scale is undefined");
    return;
  }

  const cellSize = scale(1) - scale(0);
  const maxSize = cellSize * 0.9; // Maximum size of the square, slightly smaller than cell size

  const transitionDuration = 1000;

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
    .attr("height", 0)
    .attr("fill", "#fff")
    .attr("stroke", "none");

  // Merge enter and update selections
  enter
    .merge(cells)
    .transition()
    .duration(transitionDuration)
    .attr("width", (d) => {
      const squaredValue = Math.pow(d.value, 2) / 20; // (d.value^2) / 20
      return Math.max(0, squaredValue * maxSize);
    })
    .attr("height", (d) => {
      const squaredValue = Math.pow(d.value, 2) / 20; // (d.value^2) / 20
      return Math.max(0, squaredValue * maxSize);
    })
    .attr("x", (d) => {
      const squaredValue = Math.pow(d.value, 2) / 20;
      const size = Math.max(0, squaredValue * maxSize);
      return scale(d.x) + (cellSize - size) / 2; // Center the square in the cell
    })
    .attr("y", (d) => {
      const squaredValue = Math.pow(d.value, 2) / 20;
      const size = Math.max(0, squaredValue * maxSize);
      return scale(d.y) + (cellSize - size) / 2; // Center the square in the cell
    });

  // Exit selection
  cells.exit().remove();
}
