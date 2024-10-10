import * as d3 from "d3";
import type { UnitData, Scenario, State } from "$lib/types";
import { createBackgroundGrid } from "$lib/scene";
import { gameStore, scale } from "$lib/store";
import { get } from "svelte/store";

export function updateVisualization() {
  const { currentState, gameInfo } = get(gameStore);
  const currentScale = get(scale);
  // console.log("updateVisualization called with:", { currentState, gameInfo, currentScale });

  if (!currentState || !gameInfo || !currentScale) {
    // console.log("Missing data for visualization, returning early");
    return;
  }

  // console.log("Current GameInfo in updateVisualization:", gameInfo);
  // console.log("Current State in updateVisualization:", currentState);

  const svg = d3.select<SVGSVGElement, unknown>("svg");
  createBackgroundGrid(svg, gameInfo.terrain, currentScale);

  const unitData: UnitData[] = currentState.unit_positions.map((position, i) => ({
    position,
    team: currentState.unit_teams[i],
    type: currentState.unit_types[i],
    health: currentState.unit_health[i],
    maxHealth: gameInfo.unit_type_info.unit_type_health[currentState.unit_types[i]],
    attack: currentState.prev_attack_actions[i],
  }));

  updateShapes(svg, unitData, gameInfo.unit_type_info, currentScale);
  updateHealthBars(svg, unitData, currentScale);
  updateAttackStreaks(svg, unitData, currentScale);
}

function updateShapes(
  svg: d3.Selection<SVGSVGElement, unknown, HTMLElement, unknown>,
  unitData: UnitData[],
  unitTypeInfo: Scenario["unit_type_info"],
  currentScale: d3.ScaleLinear<number, number>,
) {
  const shapes = svg.selectAll<SVGPathElement, UnitData>(".shape").data(unitData, (d, i) => i.toString());

  const duration = 750; // Adjust the duration as needed for smoother transitions

  shapes
    .enter()
    .append("path")
    .attr("class", (d) => `shape ink type-${d.type} ${d.team === 0 ? "ally" : "enemy"}`)
    .merge(shapes)
    .transition() // Start a transition
    .duration(duration) // Set the duration for the transition
    .attr("d", (d, i, nodes) => {
      const { x, y } = getPosition(d, currentScale);
      const radius = currentScale(unitTypeInfo.unit_type_radiuses[d.type]);
      return createUnitShape(d, x, y, radius);
    });

  shapes.exit().remove();
}

function updateHealthBars(
  svg: d3.Selection<SVGSVGElement, unknown, HTMLElement, unknown>,
  unitData: UnitData[],
  currentScale: d3.ScaleLinear<number, number>,
) {
  const healthBars = svg.selectAll<SVGRectElement, UnitData>(".health-bar").data(unitData, (d, i) => i.toString());

  const duration = 750; // Use the same duration for consistency

  healthBars
    .enter()
    .append("rect")
    .attr("class", "health-bar ink")
    .merge(healthBars)
    .transition() // Transition for smooth health bar updates
    .duration(duration)
    .attr("x", (d) => positionHealthBar(d, currentScale).x)
    .attr("y", (d) => positionHealthBar(d, currentScale).y)
    .attr("width", (d) => positionHealthBar(d, currentScale).width)
    .attr("height", 2);

  healthBars.exit().remove();
}
function updateAttackStreaks(
  svg: d3.Selection<SVGSVGElement, unknown, HTMLElement, unknown>,
  unitData: UnitData[],
  currentScale: d3.ScaleLinear<number, number>,
) {
  svg.selectAll(".streak").remove();

  const team0Units = unitData.filter((u) => u.team === 0);
  const team1Units = unitData.filter((u) => u.team === 1);

  for (let i = 0; i < unitData.length; i++) {
    const agent = unitData[i];
    if (agent.attack >= 5) {
      let target: UnitData | undefined;
      if (agent.team === 0) {
        const targetIndex = agent.attack - 5;
        target = team1Units[targetIndex];
      } else {
        const targetIndex = team0Units.length - 1 - (agent.attack - 5);
        target = team0Units[targetIndex];
      }

      if (target) {
        const { x1, y1, x2, y2 } = calculateStreakPositions(agent, target, currentScale);

        const line = svg
          .append("line")
          .attr("class", "streak ink")
          .attr("stroke-width", 3)
          .attr("stroke-opacity", 0.8) // Initial higher opacity for better visibility
          .attr("x1", x1)
          .attr("y1", y1)
          .attr("x2", x1)
          .attr("y2", y1);

        // Calculate a longer offset for streak length
        const streakLength = 30; // Increase the length of the streak
        const dx = x2 - x1;
        const dy = y2 - y1;
        const length = Math.sqrt(dx * dx + dy * dy);
        const unitDx = (dx / length) * streakLength;
        const unitDy = (dy / length) * streakLength;

        // Animate both ends of the line to create a moving streak effect
        line
          .transition()
          .duration(500) // Duration of the animation
          .attr("x1", x2 - unitDx)
          .attr("y1", y2 - unitDy)
          .attr("x2", x2)
          .attr("y2", y2)
          .attr("stroke-opacity", 0) // Fade out as it reaches the target
          .on("end", () => line.remove());
      }
    }
  }
}

function getPosition(d: UnitData, currentScale: d3.ScaleLinear<number, number>) {
  return { x: currentScale(d.position[0]), y: currentScale(d.position[1]) };
}

function positionHealthBar(d: UnitData, currentScale: d3.ScaleLinear<number, number>) {
  const { x, y } = getPosition(d, currentScale);
  const normalizedHealth = d.health / d.maxHealth;
  return {
    x: x - 5,
    y: y - 15,
    width: normalizedHealth * 10,
  };
}

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

function createUnitShape(d: UnitData, x: number, y: number, radius: number) {
  switch (d.type) {
    case 0:
      return `M ${x},${y} m -${radius},0 a ${radius},${radius} 0 1,0 ${radius * 2},0 a ${radius},${radius} 0 1,0 -${radius * 2},0`;
    case 1:
      return `M ${x - radius},${y - radius} h ${radius * 2} v ${radius * 2} h -${radius * 2} z`;
    case 2:
      return `M ${x},${y - radius} L ${x - radius},${y + radius} L ${x + radius},${y + radius} Z`;
    default:
      return "";
  }
}
