<script lang="ts">
    import { onMount, onDestroy } from "svelte";
    import { gameId, states, gameInfo, intervalId } from "$lib/store"; // Import the stores
    import { get } from "svelte/store";
    import { startAnimation, updateVisualization } from "$lib/plots";
    import { handleResize } from "$lib/utils"; // Import handleResize from utils.ts

    async function startGame() {
        const response = await fetch("http://localhost:8000/run", {
            method: "POST",
        });
        const data = await response.json();
        gameId.set(data.game_id);
        states.set(data.states);
        gameInfo.set(data.env_info); // Update this line to set gameInfo
        console.log(data.env_info);
        startAnimation();
        updateVisualization(100); // Add this line to immediately draw the background
    }

    onDestroy(() => {
        const currentIntervalId = get(intervalId);
        if (currentIntervalId !== null) {
            clearInterval(currentIntervalId);
        }
        if (typeof window !== "undefined") {
            window.removeEventListener("resize", handleResize);
        }
    });

    onMount(() => {
        if (typeof window !== "undefined") {
            startGame().then(() => {
                handleResize();
                window.addEventListener("resize", handleResize);
            });
        }
    });

    // Reactive statement to call handleResize when gameInfo is updated
    $: if (get(gameInfo)) {
        handleResize();
    }
</script>

<div id="simulation">
    {#if !$gameId}
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
