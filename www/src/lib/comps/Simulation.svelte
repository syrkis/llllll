<script lang="ts">
    import { onMount, onDestroy } from "svelte";
    import { gameStore, scale } from "$lib/store";
    import { createBackgroundGrid } from "$lib/scene";
    import { get } from "svelte/store";
    import * as d3 from "d3";

    let svgElement: SVGSVGElement;
    let initialRenderedTerrain: number[][] | null = null;

    function resizeSVG() {
        // Ensure that this code only runs in the browser context
        if (typeof window !== "undefined" && svgElement) {
            const width = svgElement.clientWidth;
            const height = svgElement.clientHeight;
            const newScale = d3
                .scaleLinear()
                .domain([0, 100]) // Assuming a domain of 0 to 100
                .range([0, Math.min(width, height)]);
            scale.set(newScale);
        }
    }

    function drawTerrain() {
        const { terrain } = get(gameStore);
        if (terrain && $scale) {
            const svg = d3.select<SVGSVGElement, unknown>(svgElement);
            createBackgroundGrid(svg, terrain, $scale);
            initialRenderedTerrain = terrain;
            console.log("Drawing new terrain");
        }
    }

    onMount(() => {
        // window is defined only when the component runs in a browser
        resizeSVG();
        if (typeof window !== "undefined") {
            window.addEventListener("resize", resizeSVG);
        }
    });

    onDestroy(() => {
        if (typeof window !== "undefined") {
            window.removeEventListener("resize", resizeSVG);
        }
    });

    // React to changes in the terrain or scale
    $: {
        const { terrain } = $gameStore;
        if (
            terrain &&
            $scale &&
            (!initialRenderedTerrain || JSON.stringify(initialRenderedTerrain) !== JSON.stringify(terrain))
        ) {
            drawTerrain();
        }
        // Update drawing when scale changes
        else if ($scale) {
            drawTerrain();
        }
    }
</script>

<div id="simulation">
    <svg bind:this={svgElement}></svg>
</div>

<style>
    #simulation {
        padding: 2vh;
        height: 96vh;
        width: 96vh;
    }
    svg {
        height: 100%;
        width: 100%;
    }
</style>
