<script lang="ts">
    import { onMount, onDestroy } from "svelte";
    import * as d3 from "d3";

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
    let currentStep: number = 0;
    let intervalId: ReturnType<typeof setInterval> | null = null;
    let vh: number = 0;
    let scale: d3.ScaleLinear<number, number> | null = null;

    // Define a color scale or mapping for the teams
    const teamColors = d3.scaleOrdinal(d3.schemeCategory10);

    async function startGame() {
        const response = await fetch("http://localhost:8000/run", {
            method: "POST",
        });
        const data = await response.json();
        gameId = data.game_id;
        states = data.states;
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
        }, 100);
    }

    function updateVisualization() {
        if (!states || !scale) return;

        const svg = d3.select("svg");
        const data = states.unit_positions[currentStep];
        const teams = states.unit_teams[currentStep];
        const types = states.unit_types[currentStep];

        // Remove existing shapes
        svg.selectAll("*").remove();

        // Append new shapes based on unit types
        data.forEach((position, i) => {
            const type = types[i];
            const color = teamColors(teams[i]);
            const x = scale!(position[0]);
            const y = scale!(position[1]);

            if (type === 0) {
                // Circle for type 0
                svg.append("circle")
                    .attr("cx", x)
                    .attr("cy", y)
                    .attr("r", 5)
                    .attr("fill", color)
                    .transition()
                    .duration(100)
                    .ease(d3.easeLinear)
                    .attr("cx", x)
                    .attr("cy", y);
            } else if (type === 1) {
                // Square for type 1
                svg.append("rect")
                    .attr("x", x - 5)
                    .attr("y", y - 5)
                    .attr("width", 10)
                    .attr("height", 10)
                    .attr("fill", color)
                    .transition()
                    .duration(100)
                    .ease(d3.easeLinear)
                    .attr("x", x - 5)
                    .attr("y", y - 5);
            } else if (type === 2) {
                // Triangle for type 2
                svg.append("polygon")
                    .attr("points", `${x},${y - 5} ${x - 5},${y + 5} ${x + 5},${y + 5}`)
                    .attr("fill", color)
                    .transition()
                    .duration(100)
                    .ease(d3.easeLinear)
                    .attr("points", `${x},${y - 5} ${x - 5},${y + 5} ${x + 5},${y + 5}`);
            }
        });
    }

    onDestroy(() => {
        if (intervalId !== null) {
            clearInterval(intervalId);
        }
    });

    onMount(() => {
        vh = Math.max(document.documentElement.clientHeight || 0, window.innerHeight || 0);
        scale = d3.scaleLinear().domain([0, 400]).range([0, vh]);
        updateVisualization();
        startGame();
    });
</script>

<div id="simulation">
    {#if !gameId}
        <button on:click={startGame}>run</button>
    {:else}
        <svg width="100vh" height="100vh"></svg>
    {/if}
</div>

<style>
    #simulation {
        font-family: Arial, sans-serif;
    }
    button {
        padding: 10px 20px;
        font-size: 16px;
    }
    svg {
        border: 1px solid black;
    }
</style>
