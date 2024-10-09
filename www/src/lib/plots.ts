import * as d3 from "d3";
import type { UnitData, Scenario } from "$lib/types";
import { createBackgroundGrid } from "$lib/scene";
import { states, currentStep, intervalId, scale, gameInfo } from "$lib/store";
import { get } from "svelte/store";

const INTERVAL_DURATION = 600;

// Helper functions
function getScaledPosition(d: UnitData, currentScale: d3.ScaleLinear<number, number>) {
  return {
    x: currentScale ? currentScale(d.position[0]) : 0,
    y: currentScale ? currentScale(d.position[1]) : 0,
  };
}

function createShape(d: UnitData, x: number, y: number, radius: number) {
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

function positionHealthBar(d: UnitData, currentScale: d3.ScaleLinear<number, number>) {
  const { x, y } = getScaledPosition(d, currentScale);
  const normalizedHealth = d.health / d.maxHealth;
  return {
    x: x - 5,
    y: y - 15,
    width: normalizedHealth * 10, // Multiply by 10 to keep the same visual scale
  };
}

function calculateStreakPositions(agent: UnitData, target: UnitData, currentScale: d3.ScaleLinear<number, number>) {
  const start = getScaledPosition(agent, currentScale);
  const end = getScaledPosition(target, currentScale);
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

export function startAnimation() {
  const currentIntervalId = get(intervalId);
  if (currentIntervalId !== null) {
    clearInterval(currentIntervalId);
  }
  intervalId.set(
    setInterval(() => {
      states.update((currentStates) => {
        if (currentStates) {
          currentStep.update((n) => (n < currentStates.time.length - 1 ? n + 1 : 0));
          updateVisualization(INTERVAL_DURATION);
        }
        return currentStates;
      });
    }, INTERVAL_DURATION),
  );
}

export function updateVisualization(duration: number) {
  const currentStates = get(states);
  const currentScale = get(scale);
  const currentGameInfo = get(gameInfo);
  if (!currentStates || !currentScale || !currentGameInfo) return;

  console.log("Current GameInfo in updateVisualization:", currentGameInfo);

  const svg = d3.select<SVGSVGElement, unknown>("svg");
  createBackgroundGrid(svg, currentGameInfo.terrain, currentScale);

  const step = get(currentStep);
  const unitData: UnitData[] = currentStates.unit_positions[step].map((position, i) => ({
    position,
    team: currentStates.unit_teams[step][i],
    type: currentStates.unit_types[step][i],
    health: currentStates.unit_health[step][i],
    maxHealth: currentGameInfo.unit_type_health[currentStates.unit_types[step][i]],
    attack: currentStates.prev_attack_actions[step][i],
  }));

  updateShapes(svg, unitData, currentScale, currentGameInfo, duration);
  updateHealthBars(svg, unitData, currentScale, duration);
  updateAttackStreaks(svg, unitData, currentScale, duration);
}

function updateShapes(
  svg: d3.Selection<SVGSVGElement, unknown, HTMLElement, unknown>,
  unitData: UnitData[],
  currentScale: d3.ScaleLinear<number, number>,
  gameInfo: Scenario,
  duration: number,
) {
  const shapes = svg.selectAll<SVGPathElement, UnitData>(".shape").data(unitData, (d, i) => i.toString());

  shapes
    .enter()
    .append("path")
    .attr("class", (d) => `shape ink type-${d.type} ${d.team === 0 ? "ally" : "enemy"}`)
    .merge(shapes)
    .transition()
    .duration(duration)
    .ease(d3.easeLinear)
    .attr("d", (d) => {
      const { x, y } = getScaledPosition(d, currentScale);
      const radius = currentScale(gameInfo.unit_type_radiuses[d.type]);
      return createShape(d, x, y, radius);
    });

  shapes.exit().remove();
}

function updateHealthBars(
  svg: d3.Selection<SVGSVGElement, unknown, HTMLElement, unknown>,
  unitData: UnitData[],
  currentScale: d3.ScaleLinear<number, number>,
  duration: number,
) {
  const healthBars = svg.selectAll<SVGRectElement, UnitData>(".health-bar").data(unitData, (d, i) => i.toString());

  healthBars
    .enter()
    .append("rect")
    .attr("class", "health-bar ink")
    .merge(healthBars)
    .transition()
    .duration(duration)
    .ease(d3.easeLinear)
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
  duration: number,
) {
  svg.selectAll(".streak").remove();

  const team0Units = unitData.filter((u) => u.team === 0);
  const team1Units = unitData.filter((u) => u.team === 1);

  for (let i = 0; i < unitData.length; i++) {
    const agent = unitData[i];
    if (agent.attack >= 5) {
      let target: UnitData | undefined;
      if (agent.team === 0) {
        // For team 0, attack the enemy unit at index (attack - 5)
        const targetIndex = agent.attack - 5;
        target = team1Units[targetIndex];
      } else {
        // For team 1, attack the ally unit at index (team0Units.length - 1 - (attack - 5))
        const targetIndex = team0Units.length - 1 - (agent.attack - 5);
        target = team0Units[targetIndex];
      }

      if (target) {
        const { x1, y1, x2, y2 } = calculateStreakPositions(agent, target, currentScale);
        svg
          .append("line")
          .attr("class", "streak ink")
          .attr("x1", x1)
          .attr("y1", y1)
          .attr("x2", x1)
          .attr("y2", y1)
          .attr("stroke-width", 3)
          .attr("stroke-opacity", 0.6)
          .transition()
          .duration(duration)
          .ease(d3.easeLinear)
          .attr("x2", x2)
          .attr("y2", y2)
          .attr("stroke-opacity", 0)
          .remove();
      }
    }
  }
}
