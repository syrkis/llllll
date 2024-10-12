<script lang="ts">
    import { onMount, onDestroy } from "svelte";
    import { gameStore, scale, coordinatesStore } from "$lib/store";
    import { createBackgroundGrid } from "$lib/scene";
    import { get } from "svelte/store";
    import * as d3 from "d3";

    let svgElement: SVGSVGElement;
    let initialRenderedTerrain: number[][] | null = null;
    let coordinates: { x: number; y: number; letter: string }[] = [];

    let isDragging = false;
    let dragIndex = -1;
    let dragStartX = 0;
    let dragStartY = 0;

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
                const svg = d3.select<SVGSVGElement, unknown>(svgElement);
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
        if (isDragging) return; // Prevent click action if dragging

        event.preventDefault();
        const [x, y] = d3.pointer(event, svgElement);
        const threshold = 20;

        const markerIndex = coordinates.findIndex(
            (d) => Math.abs(d.x - x) <= threshold && Math.abs(d.y - y) <= threshold,
        );

        if (markerIndex !== -1) {
            coordinates.splice(markerIndex, 1); // Remove if exists
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
        coordinatesStore.set(coordinates);
    }

    function handleMouseDown(event: MouseEvent) {
        const [x, y] = d3.pointer(event, svgElement);
        const threshold = 20;

        dragIndex = coordinates.findIndex((d) => Math.abs(d.x - x) <= threshold && Math.abs(d.y - y) <= threshold);

        if (dragIndex !== -1) {
            isDragging = true;
            dragStartX = x;
            dragStartY = y;
        }
    }

    function handleMouseMove(event: MouseEvent) {
        if (!isDragging) return;

        const [x, y] = d3.pointer(event, svgElement);
        if (dragIndex !== -1) {
            // Update the coordinate of the dragged piece
            coordinates[dragIndex].x += x - dragStartX;
            coordinates[dragIndex].y += y - dragStartY;
            dragStartX = x;
            dragStartY = y;

            drawMarkers();
        }
    }

    function handleMouseUp(event: MouseEvent) {
        if (!isDragging) return;

        isDragging = false;
        dragIndex = -1;
        coordinatesStore.set(coordinates); // Update the store when the drag ends
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
            svgElement.addEventListener("mousedown", handleMouseDown);
            svgElement.addEventListener("mousemove", handleMouseMove);
            window.addEventListener("mouseup", handleMouseUp);
            svgElement.addEventListener("click", handleSVGClick);
            document.addEventListener("contextmenu", (event) => event.preventDefault());
        }
    });

    onDestroy(() => {
        if (typeof window !== "undefined") {
            window.removeEventListener("resize", resizeSVG);
            svgElement.removeEventListener("mousedown", handleMouseDown);
            svgElement.removeEventListener("mousemove", handleMouseMove);
            window.removeEventListener("mouseup", handleMouseUp);
            svgElement.removeEventListener("click", handleSVGClick);
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
