<script lang="ts">
    import { onMount, onDestroy } from "svelte";
    import { gameStore, scale, coordinatesStore } from "$lib/store"; // Import the new coordinates store
    import { createBackgroundGrid } from "$lib/scene";
    import { get } from "svelte/store";
    import * as d3 from "d3";
    // import { sendMessage } from "$lib/api"; // Ensure this is defined if you uncomment

    let svgElement: SVGSVGElement;
    let initialRenderedTerrain: number[][] | null = null;
    let coordinates: { x: number; y: number; letter: string }[] = [];

    function resizeSVG() {
        if (typeof window !== "undefined" && svgElement) {
            const width = svgElement.clientWidth;
            const height = svgElement.clientHeight;
            const newScale = d3
                .scaleLinear()
                .domain([0, 100])
                .range([0, Math.min(width, height)]);
            scale.set(newScale);

            const terrain = get(gameStore).terrain;
            if (terrain) {
                const svg = d3.select<SVGSVGElement, unknown>(svgElement as SVGSVGElement);
                createBackgroundGrid(svg, terrain, newScale, true);
            }
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

    const chessPieces = ["♔", "♕", "♖", "♗", "♘", "♙"];
    const maxMarkers = chessPieces.length;

    function handleSVGClick(event: MouseEvent) {
        event.preventDefault();
        const [x, y] = d3.pointer(event, svgElement);
        const threshold = 20;

        const markerIndex = coordinates.findIndex(
            (d) => Math.abs(d.x - x) <= threshold && Math.abs(d.y - y) <= threshold,
        );

        if (markerIndex !== -1) {
            coordinates.splice(markerIndex, 1);
        } else {
            let newLetter = "";
            if (coordinates.length >= maxMarkers) {
                newLetter = coordinates[0].letter;
                coordinates.shift();
            } else {
                const usedLetters = new Set(coordinates.map((coord) => coord.letter));
                newLetter = chessPieces.find((piece) => !usedLetters.has(piece)) || "";
            }

            coordinates = [...coordinates, { x, y, letter: newLetter }];
        }

        drawMarkers();
        coordinatesStore.set(coordinates); // Update the store
        // sendCoordinatesToServer(); // Ensure sendMessage is imported or defined if used
    }

    function drawMarkers() {
        const svg = d3.select(svgElement);
        svg.selectAll("text").remove();

        svg.selectAll("text")
            .data(coordinates)
            .enter()
            .append("text")
            .attr("x", (d) => d.x)
            .attr("y", (d) => d.y)
            .attr("class", "ink")
            .text((d) => d.letter)
            .style("font-size", "3em");
    }

    onMount(() => {
        resizeSVG();

        if (typeof window !== "undefined") {
            window.addEventListener("resize", resizeSVG);
            svgElement.addEventListener("mousedown", handleSVGClick);
            document.addEventListener("contextmenu", (event) => event.preventDefault());
        }
    });

    onDestroy(() => {
        if (typeof window !== "undefined") {
            window.removeEventListener("resize", resizeSVG);
            svgElement.removeEventListener("mousedown", handleSVGClick);
            document.removeEventListener("contextmenu", (event) => event.preventDefault());
        }
    });

    $: {
        const { terrain } = $gameStore;
        if (
            terrain &&
            $scale &&
            (!initialRenderedTerrain || JSON.stringify(initialRenderedTerrain) !== JSON.stringify(terrain))
        ) {
            drawTerrain();
        } else if ($scale) {
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
    .ink {
        font-family: Arial, sans-serif;
    }
</style>
