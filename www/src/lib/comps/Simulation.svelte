<script lang="ts">
    import { onMount, onDestroy } from "svelte";

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
            }
        }, 100);
    }

    onDestroy(() => {
        if (intervalId !== null) {
            clearInterval(intervalId);
        }
    });

    function getTimeStepData(state: State, timeStep: number) {
        if (timeStep < 0 || timeStep >= state.time.length) {
            throw new Error("Invalid time step");
        }

        return {
            unit_positions: state.unit_positions[timeStep],
            unit_alive: state.unit_alive[timeStep],
            unit_teams: state.unit_teams[timeStep],
            unit_health: state.unit_health[timeStep],
            unit_types: state.unit_types[timeStep],
            unit_weapon_cooldowns: state.unit_weapon_cooldowns[timeStep],
            prev_movement_actions: state.prev_movement_actions[timeStep],
            prev_attack_actions: state.prev_attack_actions[timeStep],
            time: state.time[timeStep],
            terminal: state.terminal[timeStep],
        };
    }
</script>

<div id="simulation">
    {#if !gameId}
        <button on:click={startGame}>run</button>
    {:else}
        <svg width="500" height="500">
            {#if states}
                {#each getTimeStepData(states, currentStep).unit_positions as position, index}
                    <circle cx={position[0]} cy={position[1]} r="5" fill="blue" />
                {/each}
            {/if}
        </svg>
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
