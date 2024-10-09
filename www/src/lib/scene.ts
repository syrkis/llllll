import type * as d3 from "d3";
import type { SVGSelection, GridData, CellVisualizationConfig, SimulationConfig } from "$lib/types";

export function createBackgroundGrid(
  svg: d3.Selection<SVGSVGElement, unknown, HTMLElement, unknown>,
  gridData: { water: boolean[][]; walls: boolean[][]; trees: boolean[][] },
  scale: d3.ScaleLinear<number, number> | null,
) {
  if (!gridData || !scale) return;

  const { walls, water, trees } = gridData;

  const cellSize = scale(1) - scale(0);
  const solidTileSize = cellSize * 0.5;
  const waterCircleRadius = cellSize * 0.05;
  const treeSize = cellSize * 0.1;
  const offset = (cellSize - solidTileSize) / 2; // Centering offset

  // Background cells (all cells)
  svg
    .selectAll(".background-cell")
    .data(
      walls.flat().map((_, index) => ({
        x: index % walls[0].length,
        y: Math.floor(index / walls[0].length),
      })),
    )
    .join("rect")
    .attr("class", "background-cell")
    .attr("x", (d) => scale(d.x))
    .attr("y", (d) => scale(d.y))
    .attr("width", cellSize)
    .attr("height", cellSize)
    .attr("fill", "transparent");

  // Walls (solid tiles)
  svg
    .selectAll(".solid-tile")
    .data(
      walls
        .flat()
        .map((isWall, index) => ({
          isWall,
          x: index % walls[0].length,
          y: Math.floor(index / walls[0].length),
        }))
        .filter((d) => d.isWall),
    )
    .join("rect")
    .attr("class", "solid-tile ink")
    .attr("x", (d) => scale(d.x) + offset)
    .attr("y", (d) => scale(d.y) + offset)
    .attr("width", solidTileSize)
    .attr("height", solidTileSize)
    .attr("fill", "#fff")
    .attr("stroke", "none");

  // Water cells
  svg
    .selectAll(".water-cell")
    .data(
      water
        .flat()
        .map((isWater, index) => ({
          isWater,
          x: index % water[0].length,
          y: Math.floor(index / water[0].length),
        }))
        .filter((d) => d.isWater),
    )
    .join("circle")
    .attr("class", "water-cell ink")
    .attr("cx", (d) => scale(d.x) + cellSize / 2)
    .attr("cy", (d) => scale(d.y) + cellSize / 2)
    .attr("r", waterCircleRadius)
    .attr("fill", "#fff")
    .attr("stroke", "none");

  // Tree cells
  svg
    .selectAll(".tree-cell")
    .data(
      trees
        .flat()
        .map((isTree, index) => ({
          isTree,
          x: index % trees[0].length,
          y: Math.floor(index / trees[0].length),
        }))
        .filter((d) => d.isTree),
    )
    .join("circle")
    .attr("class", "tree-cell ink")
    .attr("cx", (d) => scale(d.x) + cellSize / 2)
    .attr("cy", (d) => scale(d.y) + cellSize / 2)
    .attr("r", treeSize)
    .attr("stroke", "none");
}
