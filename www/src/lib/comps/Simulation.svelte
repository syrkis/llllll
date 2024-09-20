<script lang="ts">
    import { onMount } from "svelte";

    let gameId: string | null = null;
    let observation: number[] = [];
    let reward: number = 0;
    let terminated: boolean = false;
    let truncated: boolean = false;
    let info: any = {};

    async function startGame() {
        const response = await fetch("http://localhost:8000/start_game", {
            method: "POST",
        });
        const data = await response.json();
        gameId = data.game_id;
        observation = data.initial_observation;
        info = data.info;
    }

    async function step(action: number) {
        if (!gameId) return;

        const response = await fetch(`http://localhost:8000/step/${gameId}`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ action }),
        });
        const data = await response.json();
        observation = data.observation;
        reward = data.reward;
        terminated = data.terminated;
        truncated = data.truncated;
        info = data.info;

        if (terminated || truncated) {
            endGame();
        }
    }

    async function endGame() {
        if (!gameId) return;

        await fetch(`http://localhost:8000/end_game/${gameId}`, {
            method: "DELETE",
        });
        gameId = null;
    }

    function handleKeyDown(event: KeyboardEvent) {
        if (event.key === "ArrowLeft") {
            step(0);
        } else if (event.key === "ArrowRight") {
            step(1);
        }
    }

    onMount(() => {
        window.addEventListener("keydown", handleKeyDown);
        return () => {
            window.removeEventListener("keydown", handleKeyDown);
            if (gameId) endGame();
        };
    });
</script>

<div id="simulation">
    <h1>CartPole-v1 Simulation</h1>
    {#if !gameId}
        <button on:click={startGame}>Start Game</button>
    {:else}
        <div>
            <h2>Game State</h2>
            <p>Observation: {JSON.stringify(observation)}</p>
            <p>Reward: {reward}</p>
            <p>Terminated: {terminated}</p>
            <p>Truncated: {truncated}</p>
            <p>Info: {JSON.stringify(info)}</p>
        </div>
        <div>
            <button on:click={() => step(0)}>Move Left</button>
            <button on:click={() => step(1)}>Move Right</button>
        </div>
        <button on:click={endGame}>End Game</button>
    {/if}
    <p>Use left and right arrow keys to control the cart</p>
</div>

<style>
    #simulation {
        font-family: Arial, sans-serif;
        padding: 20px;
    }
    button {
        margin: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
</style>
