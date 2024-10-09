<script lang="ts">
    import { onMount, onDestroy } from "svelte";
    import * as d3 from "d3";
    import type { ScaleLinear } from "d3-scale";
    import type { BaseType, EnterElement } from "d3-selection";
    import type { State, GridData, EnvInfo, UnitData } from "$lib/types";
    import { createBackgroundGrid } from "$lib/scene";

    let gameId: string | null = null;
    let states: State | null = null;
    // let place: string | null = null;
    let currentStep = 0;

    let gridData: GridData = { solid: [], water: [], trees: [] };
    let envInfo: EnvInfo | null = null;

    let intervalId: ReturnType<typeof setInterval> | null = null;
    let vh = 0;
    let scale: ScaleLinear<number, number> | null = null;

    async function startGame() {
        const response = await fetch("http://localhost:8000/run", {
            method: "POST",
        });
        const data = await response.json();
        gameId = data.game_id;
        states = data.states;
        // place = data.place;
        gridData = data.terrain;
        envInfo = data.env_info;
        console.log(envInfo);
        startAnimation();
        updateVisualization(); // Add this line to immediately draw the background
    }

    function startAnimation() {
        if (intervalId !== null) {
            clearInterval(intervalId);
        }
        intervalId = setInterval(() => {
            if (states) {
                if (currentStep < states.time.length - 1) {
                    currentStep += 1;
                } else {
                    currentStep = 0;
                }
                updateVisualization();
            }
        }, 1000);
    }

    function updateVisualization() {
        if (!states || !scale) return;

        const svg = d3.select<SVGSVGElement, unknown>("svg");
        createBackgroundGrid(svg, gridData, scale);

        const unitData: UnitData[] = states.unit_positions[currentStep].map((position, i) => ({
            position,
            team: states.unit_teams[currentStep][i],
            type: states.unit_types[currentStep][i],
            health: states.unit_health[currentStep][i],
            attack: states.prev_attack_actions[currentStep][i],
        }));

        // Bind data to existing shapes
        const shapes = svg.selectAll<SVGElement, UnitData>(".shape").data(unitData, (d, i) => i);

        // Update existing shapes
        shapes.each(function (d) {
            const shape = d3.select(this);
            const x = scale ? scale(d.position[0]) : 0;
            const y = scale ? scale(d.position[1]) : 0;

            if (d.type === 0) {
                shape.transition().duration(300).ease(d3.easeLinear).attr("cx", x).attr("cy", y);
            } else if (d.type === 1) {
                shape
                    .transition()
                    .duration(300)
                    .ease(d3.easeLinear)
                    .attr("x", x - 5)
                    .attr("y", y - 5);
            } else if (d.type === 2) {
                shape
                    .transition()
                    .duration(300)
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
            .attr("class", "shape ink");

        // Set initial attributes for new shapes
        newShapes.each(function (d) {
            const shape = d3.select(this);
            const x = scale ? scale(d.position[0]) : 0;
            const y = scale ? scale(d.position[1]) : 0;

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

        // Health bars
        const healthBars = svg.selectAll<SVGRectElement, UnitData>(".health-bar").data(unitData, (d, i) => i);

        healthBars
            .enter()
            .append("rect")
            .attr("class", "health-bar ink")
            .attr("x", (d) => (scale ? scale(d.position[0]) : 0) - 5)
            .attr("y", (d) => (scale ? scale(d.position[1]) : 0) - 15)
            .attr("width", 10)
            .attr("height", 2);

        healthBars
            .transition()
            .duration(300)
            .ease(d3.easeLinear)
            .attr("x", (d) => (scale ? scale(d.position[0]) : 0) - 5)
            .attr("y", (d) => (scale ? scale(d.position[1]) : 0) - 15)
            .attr("width", (d) => (d.health / 100) * 10);

        healthBars.exit().remove();

        svg.selectAll(".streak").remove();

        unitData.forEach((agent, i) => {
            if (agent.attack === 5) {
                const targetIndex = unitData.findIndex((_, idx) => unitData[idx].team !== agent.team);

                if (targetIndex !== -1) {
                    const targetData = unitData[targetIndex];

                    const x1 = scale ? scale(agent.position[0]) : 0;
                    const y1 = scale ? scale(agent.position[1]) : 0;
                    const x2 = scale ? scale(targetData.position[0]) : 0;
                    const y2 = scale ? scale(targetData.position[1]) : 0;

                    const offsetRatio = 0.05;
                    const dx = x2 - x1;
                    const dy = y2 - y1;
                    const length = Math.sqrt(dx * dx + dy * dy);

                    const offsetX = (dx / length) * offsetRatio * length;
                    const offsetY = (dy / length) * offsetRatio * length;

                    svg.append("line")
                        .attr("class", "streak ink")
                        .attr("x1", x1 + offsetX)
                        .attr("y1", y1 + offsetY)
                        .attr("x2", x1 + offsetX)
                        .attr("y2", y1 + offsetY)
                        .attr("stroke-width", 3)
                        .attr("stroke-opacity", 0.6)
                        .transition()
                        .duration(1000)
                        .ease(d3.easeLinear)
                        .attr("x2", x2 - offsetX)
                        .attr("y2", y2 - offsetY)
                        .attr("stroke-opacity", 0)
                        .remove();
                }
            }
        });
    }

    function handleResize() {
        if (typeof window !== "undefined") {
            vh = Math.max(document.documentElement.clientHeight || 0, window.innerHeight || 0) * 0.96;

            // Assuming gridData.solid contains the grid size information
            const gridSize = gridData.solid.length > 0 ? gridData.solid[0].length : 100; // Default to 100 if no data

            scale = d3.scaleLinear().domain([0, gridSize]).range([0, vh]);
            updateVisualization();
        }
    }

    onDestroy(() => {
        if (intervalId !== null) {
            clearInterval(intervalId);
        }
        if (typeof window !== "undefined") {
            window.removeEventListener("resize", handleResize);
        }
    });

    onMount(() => {
        if (typeof window !== "undefined") {
            handleResize();
            window.addEventListener("resize", handleResize);
        }
        startGame();
    });
</script>

<div id="simulation">
    {#if !gameId}
        <button on:click={startGame}>run</button>
    {:else}
        <svg />
    {/if}
</div>

<style>
    #simulation {
        padding: 2vh;
        height: 96vh;
        width: 96vh;
    }
    button {
        padding: 10px 20px;
        font-size: 16px;
    }
    svg {
        /* border: 2px solid; */
        height: 100%;
        width: 100%;
    }
</style>
