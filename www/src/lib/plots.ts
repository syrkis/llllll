import * as d3 from "d3";
import type { UnitData } from "$lib/types";
import { createBackgroundGrid } from "$lib/scene";
import { states, currentStep, intervalId, scale, gameInfo } from "$lib/store";
import { get } from "svelte/store";
import type { Selection } from "d3-selection";

const INTERVAL_DURATION = 200;

// Helper functions
function getScaledPosition(d: UnitData, currentScale: d3.ScaleLinear<number, number>) {
  return {
    x: currentScale ? currentScale(d.position[0]) : 0,
    y: currentScale ? currentScale(d.position[1]) : 0,
  };
}

function createShape(d: UnitData, x: number, y: number) {
  switch (d.type) {
    case 0:
      return `M ${x},${y} m -5,0 a 5,5 0 1,0 10,0 a 5,5 0 1,0 -10,0`;
    case 1:
      return `M ${x - 5},${y - 5} h 10 v 10 h -10 z`;
    case 2:
      return `M ${x},${y - 5} L ${x - 5},${y + 5} L ${x + 5},${y + 5} Z`;
    default:
      return "";
  }
}

function positionHealthBar(d: UnitData, currentScale: d3.ScaleLinear<number, number>) {
  const { x, y } = getScaledPosition(d, currentScale);
  return {
    x: x - 5,
    y: y - 15,
    width: (d.health / 100) * 10,
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
    attack: currentStates.prev_attack_actions[step][i],
  }));

  updateShapes(svg, unitData, currentScale, duration);
  updateHealthBars(svg, unitData, currentScale, duration);
  updateAttackStreaks(svg, unitData, currentScale, duration);
}

function updateShapes(
  svg: d3.Selection<SVGSVGElement, unknown, HTMLElement, unknown>,
  unitData: UnitData[],
  currentScale: d3.ScaleLinear<number, number>,
  duration: number,
) {
  const shapes = svg.selectAll<SVGPathElement, UnitData>(".shape").data(unitData, (d, i) => i.toString());

  shapes
    .enter()
    .append("path")
    .attr("class", (d) => `shape ink ${d.team === 0 ? "ally" : "enemy"}`)
    .merge(shapes)
    .transition()
    .duration(duration)
    .ease(d3.easeLinear)
    .attr("d", (d) => {
      const { x, y } = getScaledPosition(d, currentScale);
      return createShape(d, x, y);
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

  for (const agent of unitData) {
    if (agent.attack === 5) {
      const target = unitData.find((u) => u.team !== agent.team);
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
