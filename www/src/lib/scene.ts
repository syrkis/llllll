import type * as d3 from "d3";
import { easeCubicInOut } from "d3";
import { transitionDurations } from "$lib/store";
import { TerrainType } from "$lib/types";
import { get } from "svelte/store";

interface TerrainCell {
  value: TerrainType;
  x: number;
  y: number;
}

// Helper function to convert TerrainType to class name
function getTerrainClass(terrainType: TerrainType): string {
  switch (terrainType) {
    case TerrainType.Empty:
      return "empty";
    case TerrainType.Water:
      return "water";
    case TerrainType.Trees:
      return "trees";
    case TerrainType.Walls:
      return "walls";
    default:
      return "empty";
  }
}
// scene.ts
export function createBackgroundGrid(
  svg: d3.Selection<SVGSVGElement, unknown, HTMLElement, unknown>,
  terrainMatrix: TerrainType[][] | null,
  scale: d3.ScaleLinear<number, number> | null,
  isResize = false,
) {
  if (!terrainMatrix || !scale) {
    console.warn("Terrain matrix or scale is undefined");
    return;
  }

  // Create local non-null scale reference
  const scaleNonNull = scale;
  const duration = get(transitionDurations).terrain;
  const cellSize = scaleNonNull(1) - scaleNonNull(0);

  // Log the first few cells of terrain data
  console.log("Sample terrain data:", terrainMatrix[0].slice(0, 5));

  const terrainSizeMap = new Map<TerrainType, number>([
    [TerrainType.Empty, 0.01],
    [TerrainType.Water, 0.3],
    [TerrainType.Trees, 0.6],
    [TerrainType.Walls, 0.5],
    [TerrainType.Solid, 0.5],
  ]);

  function getSizeMultiplier(value: TerrainType): number {
    const multiplier = terrainSizeMap.get(value);
    if (multiplier === undefined) {
      console.error("Unknown terrain type:", value);
      return 0.01;
    }
    return multiplier;
  }

  function calculateSize(d: TerrainCell): number {
    console.log("Calculating size for cell:", d);
    const multiplier = getSizeMultiplier(d.value);
    return cellSize * multiplier;
  }

  function calculateX(d: TerrainCell): number {
    const size = calculateSize(d);
    return scaleNonNull(d.x) + (cellSize - size) / 2;
  }

  function calculateY(d: TerrainCell): number {
    const size = calculateSize(d);
    return scaleNonNull(100 - d.y) - cellSize + (cellSize - size) / 2;
  }

  const terrainData: TerrainCell[] = terrainMatrix.flat().map((value, index) => {
    const cell = {
      value,
      x: index % terrainMatrix[0].length,
      y: Math.floor(index / terrainMatrix[0].length),
    };
    if (index === 0) {
      console.log("First cell:", cell);
      console.log("First cell value type:", typeof cell.value);
    }
    return cell;
  });

  // Rest of the code remains the same...
  const cells = svg.selectAll<SVGRectElement, TerrainCell>(".terrain-cell").data(terrainData, (d) => `${d.x}-${d.y}`);

  // Enter selection
  const enter = cells
    .enter()
    .append("rect")
    .attr("class", (d) => `terrain-cell ink ${getTerrainClass(d.value)}`)
    .attr("x", calculateX)
    .attr("y", calculateY)
    .attr("width", calculateSize)
    .attr("height", calculateSize);

  // Update selection
  cells
    .transition()
    .duration(duration)
    .ease(easeCubicInOut)
    .attr("class", (d) => `terrain-cell ink ${getTerrainClass(d.value)}`)
    .attr("width", calculateSize)
    .attr("height", calculateSize)
    .attr("x", calculateX)
    .attr("y", calculateY);

  // Exit selection
  cells.exit().remove();
}
