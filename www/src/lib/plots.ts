import * as d3 from "d3";
import type { UnitData } from "$lib/types";
import { createBackgroundGrid } from "$lib/scene";
import { states, currentStep, intervalId, scale, gameInfo } from "$lib/store"; // Import gameInfo instead of gridData
import { get } from "svelte/store";
import type { Selection, EnterElement } from "d3-selection";

const INTERVAL_DURATION = 300; // Define the interval duration

export function startAnimation() {
  const currentIntervalId = get(intervalId);
  if (currentIntervalId !== null) {
    clearInterval(currentIntervalId);
  }
  intervalId.set(
    setInterval(() => {
      states.update((currentStates) => {
        if (currentStates) {
          if (get(currentStep) < currentStates.time.length - 1) {
            currentStep.update((n) => n + 1);
          } else {
            currentStep.set(0);
          }
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
  const currentGameInfo = get(gameInfo); // Get the current gameInfo
  if (!currentStates || !currentScale || !currentGameInfo) return;

  console.log("Current GameInfo in updateVisualization:", currentGameInfo); // Add detailed logging

  const svg = d3.select<SVGSVGElement, unknown>("svg") as unknown as Selection<SVGSVGElement, unknown, null, undefined>;
  createBackgroundGrid(svg, currentGameInfo.terrain, currentScale); // Use terrain from gameInfo

  const unitData: UnitData[] = currentStates.unit_positions[get(currentStep)].map((position, i) => ({
    position,
    team: currentStates.unit_teams[get(currentStep)][i],
    type: currentStates.unit_types[get(currentStep)][i],
    health: currentStates.unit_health[get(currentStep)][i],
    attack: currentStates.prev_attack_actions[get(currentStep)][i],
  }));

  updateShapes(svg, unitData, currentScale, duration);
  updateHealthBars(svg, unitData, currentScale, duration);
  updateAttackStreaks(svg, unitData, currentScale, duration);
}

function updateShapes(
  svg: Selection<SVGSVGElement, unknown, null, undefined>,
  unitData: UnitData[],
  currentScale: d3.ScaleLinear<number, number>,
  duration: number,
) {
  // Bind data to existing shapes
  const shapes = svg.selectAll<SVGElement, UnitData>(".shape").data(unitData, (d, i) => i);

  // Update existing shapes
  shapes.each(function (d) {
    const shape = d3.select(this);
    const x = currentScale ? currentScale(d.position[0]) : 0;
    const y = currentScale ? currentScale(d.position[1]) : 0;

    // Update class based on team
    shape.classed("ally", d.team === 0).classed("enemy", d.team === 1);

    if (d.type === 0) {
      shape.transition().duration(duration).ease(d3.easeLinear).attr("cx", x).attr("cy", y);
    } else if (d.type === 1) {
      shape
        .transition()
        .duration(duration)
        .ease(d3.easeLinear)
        .attr("x", x - 5)
        .attr("y", y - 5);
    } else if (d.type === 2) {
      shape
        .transition()
        .duration(duration)
        .ease(d3.easeLinear)
        .attr("points", `${x},${y - 5} ${x - 5},${y + 5} ${x + 5},${y + 5}`);
    }
  });

  // Append new shapes
  const newShapes = shapes
    .enter()
    .append(function (this: EnterElement, d: UnitData): SVGElement {
      if (d.type === 0) {
        return document.createElementNS("http://www.w3.org/2000/svg", "circle");
      }
      return document.createElementNS("http://www.w3.org/2000/svg", "rect");
    })
    .attr("class", (d) => `shape ink ${d.team === 0 ? "ally" : "enemy"}`);

  // Set initial attributes for new shapes
  newShapes.each(function (d) {
    const shape = d3.select(this);
    const x = currentScale ? currentScale(d.position[0]) : 0;
    const y = currentScale ? currentScale(d.position[1]) : 0;

    if (d.type === 0) {
      shape.attr("cx", x).attr("cy", y).attr("r", 5);
    } else if (d.type === 1) {
      shape
        .attr("x", x - 5)
        .attr("y", y - 5)
        .attr("width", 10)
        .attr("height", 10);
    } else if (d.type === 2) {
      shape.attr("points", `${x},${y - 5} ${x - 5},${y + 5} ${x + 5},${y + 5}`);
    }
  });
  shapes.exit().remove();
}

function updateHealthBars(
  svg: Selection<SVGSVGElement, unknown, null, undefined>,
  unitData: UnitData[],
  currentScale: d3.ScaleLinear<number, number>,
  duration: number,
) {
  // Health bars
  const healthBars = svg.selectAll<SVGRectElement, UnitData>(".health-bar").data(unitData, (d, i) => i);

  healthBars
    .enter()
    .append("rect")
    .attr("class", "health-bar ink")
    .attr("x", (d) => (currentScale ? currentScale(d.position[0]) : 0) - 5)
    .attr("y", (d) => (currentScale ? currentScale(d.position[1]) : 0) - 15)
    .attr("width", 10)
    .attr("height", 2);

  healthBars
    .transition()
    .duration(duration)
    .ease(d3.easeLinear)
    .attr("x", (d) => (currentScale ? currentScale(d.position[0]) : 0) - 5)
    .attr("y", (d) => (currentScale ? currentScale(d.position[1]) : 0) - 15)
    .attr("width", (d) => (d.health / 100) * 10);

  healthBars.exit().remove();
}

function updateAttackStreaks(
  svg: Selection<SVGSVGElement, unknown, null, undefined>,
  unitData: UnitData[],
  currentScale: d3.ScaleLinear<number, number>,
  duration: number,
) {
  svg.selectAll(".streak").remove();

  unitData.forEach((agent, i) => {
    if (agent.attack === 5) {
      const targetIndex = unitData.findIndex((_, idx) => unitData[idx].team !== agent.team);

      if (targetIndex !== -1) {
        const targetData = unitData[targetIndex];

        const x1 = currentScale ? currentScale(agent.position[0]) : 0;
        const y1 = currentScale ? currentScale(agent.position[1]) : 0;
        const x2 = currentScale ? currentScale(targetData.position[0]) : 0;
        const y2 = currentScale ? currentScale(targetData.position[1]) : 0;

        const offsetRatio = 0.05;
        const dx = x2 - x1;
        const dy = y2 - y1;
        const length = Math.sqrt(dx * dx + dy * dy);

        const offsetX = (dx / length) * offsetRatio * length;
        const offsetY = (dy / length) * offsetRatio * length;

        svg
          .append("line")
          .attr("class", "streak ink")
          .attr("x1", x1 + offsetX)
          .attr("y1", y1 + offsetY)
          .attr("x2", x1 + offsetX)
          .attr("y2", y1 + offsetY)
          .attr("stroke-width", 3)
          .attr("stroke-opacity", 0.6)
          .transition()
          .duration(duration)
          .ease(d3.easeLinear)
          .attr("x2", x2 - offsetX)
          .attr("y2", y2 - offsetY)
          .attr("stroke-opacity", 0)
          .remove();
      }
    }
  });
}
