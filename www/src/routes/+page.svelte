<script lang="ts">
    import { onMount } from "svelte";
    import Simulation from "$lib/comps/Simulation.svelte";
    import Controller from "$lib/comps/Controller.svelte";
    import Visualizer from "$lib/comps/Visualizer.svelte";

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
    <div id="sim" class="section"><Simulation /></div>
    <!-- <div id="vis" class="section"><Visualizer /></div> -->
    <div id="con" class="section"><Controller /></div>
</div>

<style>
    .container {
        display: grid;
        grid-template-columns: 100vh 1fr;
        grid-template-rows: 1fr;
        /* add spacing between the cells*/
        /* gap: 1rem; */
        /* height: calc(100vh - 4px); */
        width: 100vw;
    }

    .section {
        /* add radius to section borders*/
        /* border: 2px solid white; */
        /* border-radius: 0.5rem; */
    }

    #sim {
        grid-column: 1;
        /* grid-row: 1 / span 2; */
        height: 100%;
        width: 100vh;
    }

    /* #vis {
        grid-column: 2;
        grid-row: 1;
    } */

    #con {
        grid-column: 2;
    }

    .section {
        overflow: auto;
    }
</style>
