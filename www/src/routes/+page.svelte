<script lang="ts">
    import Simulation from "$lib/comps/Simulation.svelte";
    import Controller from "$lib/comps/Controller.svelte";
</script>

<div class="container">
    <div id="sim"><Simulation /></div>
    <div id="con"><Controller /></div>
</div>

<style>
    .container {
        display: flex;
        width: 100vw;
        height: 100vh; /* Set container to full height */
        flex-direction: column; /* Stack columns vertically */
    }

    /* For landscape orientation (wider than taller) */
    @media (min-aspect-ratio: 1/1) {
        .container {
            flex-direction: row; /* Stack rows horizontally */
        }
    }
</style>
