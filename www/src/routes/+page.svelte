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
        display: grid;
        grid-template-columns: 100vh 1fr;
        grid-template-rows: 1fr;
        width: 100vw;
    }

    #sim {
        grid-column: 1;
        height: 100%;
        width: 100%;
    }

    #con {
        grid-column: 2;
    }
</style>
