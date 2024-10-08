import type * as d3 from "d3";
import type { SVGSelection, GridData, CellVisualizationConfig, SimulationConfig } from "$lib/types";

export function createBackgroundGrid(
  svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
  gridData: GridData,
  scale: d3.ScaleLinear<number, number> | null,
) {
  if (!gridData || !scale) return;

  const { solid, water, trees } = gridData;

  const cellSize = scale(1) - scale(0);
  const solidTileSize = cellSize * 0.5;
  const waterCircleRadius = cellSize * 0.2;
  const treeSize = cellSize * 0.1;
  const offset = (cellSize - solidTileSize) / 2; // Centering offset

  svg
    .selectAll(".background-cell")
    .data(
      solid.flat().map((isSolid, index) => ({
        isSolid,
        x: index % solid[0].length,
        y: Math.floor(index / solid[0].length),
      })),
    )
    .join("rect")
    .attr("class", "background-cell")
    .attr("x", (d) => scale(d.x))
    .attr("y", (d) => scale(d.y))
    .attr("width", cellSize)
    .attr("height", cellSize)
    .attr("fill", "transparent");

  svg
    .selectAll(".solid-tile")
    .data(
      solid
        .flat()
        .map((isSolid, index) => ({
          isSolid,
          x: index % solid[0].length,
          y: Math.floor(index / solid[0].length),
        }))
        .filter((d) => d.isSolid),
    )
    .join("rect")
    .attr("class", "solid-tile ink")
    .attr("x", (d) => scale(d.x) + offset)
    .attr("y", (d) => scale(d.y) + offset)
    .attr("width", solidTileSize)
    .attr("height", solidTileSize)
    .attr("fill", "#fff")
    .attr("stroke", "none");

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
    // .attr("fill", "#fff")
    .attr("stroke", "none");
}
