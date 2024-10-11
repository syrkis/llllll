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

    // Updated set of symbols
    const symbols = [
        "♔",
        "♕",
        "♖",
        "♗",
        "♘",
        "♙", // Chess pieces
        "♥",
        "♠",
        "♦",
        "♣", // Card suits
        "⚀",
        "⚁",
        "⚂",
        "⚃",
        "⚄", // Die faces
    ];

    function handleSVGClick(event: MouseEvent) {
        event.preventDefault();

        if (event.button === 2) {
            // Right-click to remove markers
            const [x, y] = d3.pointer(event, svgElement);
            coordinates = coordinates.filter((d) => {
                const threshold = 10;
                return Math.abs(d.x - x) > threshold || Math.abs(d.y - y) > threshold;
            });
            drawMarkers();
        } else if (event.button === 0) {
            // Left-click to add markers
            const [x, y] = d3.pointer(event, svgElement);

            // Cycle through symbols instead of letters
            const symbol = symbols[coordinates.length % symbols.length];
            coordinates = [...coordinates, { x: x, y: y, letter: symbol }];
            drawMarkers();

            // Send coordinates to server
            sendCoordinatesToServer();
        }
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
            .style("font-size", "2em");
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
