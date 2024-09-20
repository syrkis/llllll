<script lang="ts">
    import { onMount } from "svelte";
    import Simulation from "$lib/comps/Simulation.svelte";
    import Controller from "$lib/comps/Controller.svelte";

    let message = "Loading...";
    let error: string | null = null;

    async function fetchData() {
        try {
            const response = await fetch("http://localhost:8000/");
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            message = data.Hello;
        } catch (e: any) {
            console.error("There was a problem with the fetch operation: " + e.message);
            error = e.message;
        }
    }

    onMount(fetchData);
</script>

<div class="container">
    <div class="section"><Simulation /></div>
    <div class="section"><Controller /></div>
</div>

<style>
    .container {
        display: flex;
        flex-direction: row;
        justify-content: space-between;
        width: 100vw;
        height: 100vh;
    }

    :global(#simulation) {
        background-color: red;
        width: 100vh;
        height: 100vh;
    }
    /*controler should have the rest of the width, adjusting to the simuatlion*/
    :global(#controller) {
        background-color: blue;
        width: calc(100vw - 100vh);
        height: 100vh;
    }
</style>
