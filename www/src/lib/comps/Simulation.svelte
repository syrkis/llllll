<script lang="ts">
    import { onMount, onDestroy } from "svelte";
    import { gameStore, scale } from "$lib/store";
    import { createBackgroundGrid } from "$lib/scene";
    import { get } from "svelte/store";
    import * as d3 from "d3";
    import { sendMessage } from "$lib/api"; // Import your API function

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

    // Define your chess pieces
    const chessPieces = ["♔", "♕", "♖", "♗", "♘", "♙"];
    const maxMarkers = chessPieces.length; // Maximum number of markers

    function handleSVGClick(event: MouseEvent) {
        event.preventDefault();

        const [x, y] = d3.pointer(event, svgElement);

        // Check if the click is on an existing marker
        const threshold = 20; // Tolerance for clicking on a marker
        const markerIndex = coordinates.findIndex(
            (d) => Math.abs(d.x - x) <= threshold && Math.abs(d.y - y) <= threshold,
        );

        if (markerIndex !== -1) {
            // If a marker is found within the click threshold, remove it
            coordinates.splice(markerIndex, 1);
        } else {
            // Add a new marker if the click is not on an existing marker
            let newLetter;
            if (coordinates.length >= maxMarkers) {
                // If max markers are reached, remove the first and reuse its marker type
                newLetter = coordinates[0].letter;
                coordinates.shift();
            } else {
                // Find the first unused chess piece
                const usedLetters = new Set(coordinates.map((coord) => coord.letter));
                newLetter = chessPieces.find((piece) => !usedLetters.has(piece));
            }

            // Add the new marker
            coordinates = [...coordinates, { x, y, letter: newLetter }];
        }

        drawMarkers();
        sendCoordinatesToServer();
    }

    function drawMarkers() {
        const svg = d3.select(svgElement);

        // Remove existing text elements
        svg.selectAll("text").remove();

        // Append new text elements
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

    function sendCoordinatesToServer() {
        sendMessage({ coordinates })
            .then((response) => console.log("Coordinates sent:", response))
            .catch((error) => console.error("Error sending coordinates:", error));
    }

    onMount(() => {
        resizeSVG();

        if (typeof window !== "undefined") {
            window.addEventListener("resize", resizeSVG);
            svgElement.addEventListener("mousedown", handleSVGClick);

            // Prevent right-click context menu on the entire document
            document.addEventListener("contextmenu", (event) => event.preventDefault());
        }
    });

    onDestroy(() => {
        if (typeof window !== "undefined") {
            window.removeEventListener("resize", resizeSVG);
            svgElement.removeEventListener("mousedown", handleSVGClick);

            // Remove context menu listener
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
