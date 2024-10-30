import * as d3 from "d3";
import type { UnitData, Scenario, State } from "$lib/types";
import { createBackgroundGrid } from "$lib/scene";
import { gameStore, scale, transitionDurations } from "$lib/store";
import { get } from "svelte/store";

let terrainDrawn = false; // Track whether terrain has been drawn

export function updateVisualization() {
  const { currentState, gameInfo } = get(gameStore);
  const currentScale = get(scale);

  if (!currentState || !gameInfo || !currentScale) {
    return;
  }

  const svg = d3.select<SVGSVGElement, unknown>("svg");
  if (!terrainDrawn) {
    createBackgroundGrid(svg, gameInfo.terrain, currentScale);
    terrainDrawn = true;
  }

  // Filter unit data to include only living units
  const unitData: UnitData[] = currentState.unit_positions
    .map((position, i) => ({
      position,
      team: currentState.unit_teams[i],
      type: currentState.unit_types[i],
      health: currentState.unit_health[i],
      maxHealth: gameInfo.unit_type_info.unit_type_health[currentState.unit_types[i]],
      attack: currentState.prev_attack_actions[i],
    }))
    .filter((unit, i) => currentState.unit_alive[i] > 0); // Filter by alive status

  updateShapes(svg, unitData, gameInfo.unit_type_info, currentScale);
  // updateHealthBars(svg, unitData, currentScale);
  updateAttackStreaks(svg, unitData, currentScale);
}

function updateShapes(
  svg: d3.Selection<SVGSVGElement, unknown, HTMLElement, any>,
  unitData: UnitData[],
  unitTypeInfo: Scenario["unit_type_info"],
  currentScale: d3.ScaleLinear<number, number>,
) {
  const shapes = svg.selectAll<SVGPathElement, UnitData>(".shape").data(unitData, (d: UnitData, i) => i.toString());
  // Duration of the entering and exiting transition
  const duration = get(transitionDurations).shape;

  const enterShapes = shapes
    .enter()
    .append("path")
    .attr("class", (d) => `shape ink type-${d.type} ${d.team === 0 ? "ally" : "enemy"}`)
    .attr("d", (d) => {
      const { x, y } = getPosition(d, currentScale);
      // Start with zero radius for entering elements
      return createUnitShape(d, x, y, 0);
    });

  // Transition new shapes to their actual size
  enterShapes
    .transition()
    .duration(duration)
    .attr("d", (d) => {
      const { x, y } = getPosition(d, currentScale);
      const radius = currentScale(unitTypeInfo.unit_type_radiuses[d.type]);
      return createUnitShape(d, x, y, radius);
    });

  // Update existing shapes
  shapes
    .merge(enterShapes)
    .transition()
    .duration(duration)
    .attr("d", (d) => {
      const { x, y } = getPosition(d, currentScale);
      const radius = currentScale(unitTypeInfo.unit_type_radiuses[d.type]);
      return createUnitShape(d, x, y, radius);
    });

  // Transition exiting shapes to zero size, then remove
  shapes
    .exit()
    .transition()
    .duration(duration)
    .attr("d", (d: UnitData) => {
      const { x, y } = getPosition(d, currentScale);
      return createUnitShape(d, x, y, 0);
    })
    .remove();
}

// function updateHealthBars(
//   svg: d3.Selection<SVGSVGElement, unknown, HTMLElement, unknown>,
//   unitData: UnitData[],
//   currentScale: d3.ScaleLinear<number, number>,
// ) {
//   const healthBars = svg.selectAll<SVGRectElement, UnitData>(".health-bar").data(unitData, (d, i) => i.toString());

//   const duration = 300;

//   // Efficiently handle both enter and update selections
//   healthBars
//     .enter()
//     .append("rect")
//     .attr("class", "health-bar ink")
//     // Remove the transition from the initial entry, setting positions immediately
//     .attr("x", (d) => positionHealthBar(d, currentScale).x)
//     .attr("y", (d) => positionHealthBar(d, currentScale).y)
//     .attr("width", (d) => positionHealthBar(d, currentScale).width)
//     .attr("height", 2)
//     .merge(healthBars)
//     .transition()
//     .duration(duration) // Apply transitions only to the update selections
//     .attr("x", (d) => positionHealthBar(d, currentScale).x)
//     .attr("y", (d) => positionHealthBar(d, currentScale).y)
//     .attr("width", (d) => positionHealthBar(d, currentScale).width);

//   healthBars.exit().remove();
// }

function updateAttackStreaks(
  svg: d3.Selection<SVGSVGElement, unknown, HTMLElement, unknown>,
  unitData: UnitData[],
  currentScale: d3.ScaleLinear<number, number>,
) {
  svg.selectAll(".streak").remove();

  const team0Units = unitData.filter((u) => u.team === 0);
  const team1Units = unitData.filter((u) => u.team === 1);
  const duration = get(transitionDurations).streak;

  for (const agent of team0Units) {
    // Use for...of instead of forEach
    if (agent.attack >= 5) {
      const target = team1Units[agent.attack - 5];
      if (target) {
        const { x1, y1, x2, y2 } = calculateStreakPositions(agent, target, currentScale);

        svg
          .append("line")
          .attr("class", "streak ink")
          .attr("stroke-width", 3)
          .attr("stroke-opacity", 0.8)
          .attr("x1", x1)
          .attr("y1", y1)
          .attr("x2", x1)
          .attr("y2", y1)
          .transition()
          .duration(duration)
          .attr("x1", x2)
          .attr("y1", y2)
          .attr("stroke-opacity", 0)
          .remove();
      }
    }
  }
}

function getPosition(d: UnitData, currentScale: d3.ScaleLinear<number, number>) {
  // Flip the y coordinate by subtracting from 100 (assuming 100x100 grid)
  return {
    x: currentScale(d.position[0]),
    y: currentScale(100 - d.position[1]), // HARD CODED FLIP (ASSUMING SIZE IS 100)
  };
}

// function positionHealthBar(d: UnitData, currentScale: d3.ScaleLinear<number, number>) {
//   const position = getPosition(d, currentScale);
//   return {
//     x: position.x - 5,
//     y: position.y - 15,
//     width: (d.health / d.maxHealth) * 10,
//   };
// }

function calculateStreakPositions(agent: UnitData, target: UnitData, currentScale: d3.ScaleLinear<number, number>) {
  const start = getPosition(agent, currentScale);
  const end = getPosition(target, currentScale);
  const offsetRatio = 0.05;
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const length = Math.sqrt(dx * dx + dy * dy);
  const offsetX = (dx / length) * offsetRatio * length;
  const offsetY = (dy / length) * offsetRatio * length;

  return {
    x1: start.x + offsetX,
    y1: start.y + offsetY,
    x2: end.x - offsetX,
    y2: end.y - offsetY,
  };
}
const UNIT_TYPES = {
  CIRCLE: 0, // e.g., for ranged units
  SQUARE: 2, // e.g., for melee units
  TRIANGLE: 1, // e.g., for special units
} as const;

function createUnitShape(d: UnitData, x: number, y: number, radius: number): string {
  // change y to be 100 - y
  switch (d.type) {
    case UNIT_TYPES.CIRCLE:
      // Circle
      return `M ${x},${y} m -${radius},0
              a ${radius},${radius} 0 1,0 ${radius * 2},0
              a ${radius},${radius} 0 1,0 -${radius * 2},0`;

    case UNIT_TYPES.SQUARE:
      // Square
      return `M ${x - radius},${y - radius}
              h ${radius * 2} v ${radius * 2}
              h -${radius * 2} z`;

    case UNIT_TYPES.TRIANGLE:
      // Triangle
      return `M ${x},${y - radius}
              L ${x - radius},${y + radius}
              L ${x + radius},${y + radius} Z`;

    default:
      return "";
  }
}
