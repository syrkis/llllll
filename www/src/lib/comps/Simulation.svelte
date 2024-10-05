<script lang="ts">
    import { onMount, onDestroy } from "svelte";
    import * as d3 from "d3";
    import type { ScaleOrdinal } from "d3-scale";
    import type { BaseType, EnterElement } from "d3-selection";

    interface State {
        unit_positions: number[][][];
        unit_alive: number[][];
        unit_teams: number[][];
        unit_health: number[][];
        unit_types: number[][];
        unit_weapon_cooldowns: number[][];
        prev_movement_actions: number[][][];
        prev_attack_actions: number[][];
        time: number[];
        terminal: number[];
    }

    let gameId: string | null = null;
    let states: State | null = null;
    let place: string | null = null;
    let currentStep: number = 0;
    let width: number = 0; // map width
    let intervalId: ReturnType<typeof setInterval> | null = null;
    let vh: number = 0;
    let scale: d3.ScaleLinear<number, number> | null = null;

    // Define a color scale or mapping for the two teams (gray scale and green)
    const teamColors: ScaleOrdinal<number, string> = d3
        .scaleOrdinal<number, string>()
        .domain([0, 1])
        .range(["#888", "#4b5320"]);
    async function startGame() {
        const response = await fetch("http://localhost:8000/run", {
            method: "POST",
        });
        const data = await response.json();
        gameId = data.game_id;
        states = data.states;
        place = data.place;
        width = data.width;
        startAnimation();
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
                    currentStep = 0; // Restart the animation
                }
                updateVisualization();
            }
        }, 1000);
    }

    function updateVisualization() {
        if (!states || !scale) return;

        const svg = d3.select("svg");
        const data = states.unit_positions[currentStep];
        const teams = states.unit_teams[currentStep];
        const types = states.unit_types[currentStep];
        const attacks = states.prev_attack_actions[currentStep];

        // Bind data to existing shapes
        const shapes = svg.selectAll(".shape").data(data, (d, i) => i);

        // Update existing shapes
        shapes.each(function (d, i) {
            const shape = d3.select(this);
            const type = types[i];
            const color = teamColors(teams[i]);
            const x = scale!(d[0]);
            const y = scale!(d[1]);

            if (type === 0) {
                shape.transition().duration(600).ease(d3.easeLinear).attr("cx", x).attr("cy", y).attr("fill", color);
            } else if (type === 1) {
                shape
                    .transition()
                    .duration(1000)
                    .ease(d3.easeLinear)
                    .attr("x", x - 5)
                    .attr("y", y - 5)
                    .attr("fill", color);
            } else if (type === 2) {
                shape
                    .transition()
                    .duration(1000)
                    .ease(d3.easeLinear)
                    .attr("points", `${x},${y - 5} ${x - 5},${y + 5} ${x + 5},${y + 5}`)
                    .attr("fill", color);
            }
        });

        // Append new shapes
        const newShapes = shapes
            .enter()
            .append(function (this: EnterElement, d: number[], i: number): SVGElement {
                const type = types[i];
                if (type === 0) {
                    return document.createElementNS("http://www.w3.org/2000/svg", "circle");
                } else if (type === 1) {
                    return document.createElementNS("http://www.w3.org/2000/svg", "rect");
                } else {
                    // Default to polygon if type is neither 0 nor 1
                    return document.createElementNS("http://www.w3.org/2000/svg", "polygon");
                }
            } as (
                this: EnterElement,
                datum: number[],
                index: number,
                groups: EnterElement[] | ArrayLike<EnterElement>,
            ) => SVGElement)
            .attr("class", "shape");

        // Set initial attributes for new shapes
        newShapes.each(function (d, i) {
            const shape = d3.select(this);
            const type = types[i];
            const color = teamColors(teams[i]);
            const x = scale!(d[0]);
            const y = scale!(d[1]);

            if (type === 0) {
                shape.attr("cx", x).attr("cy", y).attr("r", 5).attr("fill", color);
            } else if (type === 1) {
                shape
                    .attr("x", x - 5)
                    .attr("y", y - 5)
                    .attr("width", 10)
                    .attr("height", 10)
                    .attr("fill", color);
            } else if (type === 2) {
                shape.attr("points", `${x},${y - 5} ${x - 5},${y + 5} ${x + 5},${y + 5}`).attr("fill", color);
            }
        });
        shapes.exit().remove();

        // Update streaks
        svg.selectAll(".streak").remove(); // Clear existing streaks

        // Add new streaks
        data.forEach((agent, i) => {
            const attack = attacks[i];
            if (attack >= 5) {
                const targetIndex = attack - 5;
                if (targetIndex >= 0 && targetIndex < data.length) {
                    const targetData = data[targetIndex];
                    const x1 = scale!(agent[0]);
                    const y1 = scale!(agent[1]);
                    const x2 = scale!(targetData[0]);
                    const y2 = scale!(targetData[1]);
                    const color = teamColors(teams[i]); // Shooter's color

                    svg.append("line")
                        .attr("class", "streak")
                        .attr("x1", x1)
                        .attr("y1", y1)
                        .attr("x2", x1) // Start at the shooter's position
                        .attr("y2", y1)
                        .attr("stroke", color)
                        .attr("stroke-width", 2)
                        .attr("stroke-opacity", 0.6)
                        .transition()
                        .duration(1000)
                        .ease(d3.easeLinear)
                        .attr("x2", x2)
                        .attr("y2", y2)
                        .attr("stroke-opacity", 0)
                        .remove();
                }
            }
        });
    }

    onDestroy(() => {
        if (intervalId !== null) {
            clearInterval(intervalId);
        }
    });

    onMount(() => {
        vh = Math.max(document.documentElement.clientHeight || 0, window.innerHeight || 0) * 0.96;
        scale = d3.scaleLinear().domain([0, 400]).range([0, vh]);
        updateVisualization();
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
        border: 2px solid #888;
        height: 100%;
        width: 100%;
    }
</style>
