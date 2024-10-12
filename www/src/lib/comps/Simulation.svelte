<script lang="ts">
    import { onMount, onDestroy } from "svelte";
    import { gameStore, scale, coordinatesStore } from "$lib/store";
    import { createBackgroundGrid } from "$lib/scene";
    import { get } from "svelte/store";
    import * as d3 from "d3";

    let svgElement: SVGSVGElement;
    let initialRenderedTerrain: number[][] | null = null;

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

    const threshold = 20; // tolerance for selecting a piece

    function handleSVGClick(event: MouseEvent) {
        if (isDragging) return; // Prevent click action if dragging

        event.preventDefault();
        const [x, y] = d3.pointer(event, svgElement);

        coordinatesStore.update((pieces) => {
            let pieceIndex = pieces.findIndex(
                (p) => p.active && Math.abs(p.x - x) <= threshold && Math.abs(p.y - y) <= threshold,
            );
            if (pieceIndex !== -1) {
                // Toggle active status
                pieces[pieceIndex].active = false;
            } else {
                pieceIndex = pieces.findIndex((p) => !p.active); // Find the first inactive piece
                if (pieceIndex !== -1) {
                    // Place inactive piece
                    pieces[pieceIndex].x = x;
                    pieces[pieceIndex].y = y;
                    pieces[pieceIndex].active = true;
                }
            }
            return pieces;
        });

        drawMarkers();
    }

    function handleMouseDown(event: MouseEvent) {
        const [x, y] = d3.pointer(event, svgElement);

        coordinatesStore.update((pieces) => {
            dragIndex = pieces.findIndex(
                (p) => p.active && Math.abs(p.x - x) <= threshold && Math.abs(p.y - y) <= threshold,
            );

            if (dragIndex !== -1) {
                isDragging = true;
                dragStartX = x;
                dragStartY = y;
            }
            return pieces;
        });
    }

    function handleMouseMove(event: MouseEvent) {
        if (!isDragging) return;

        const [x, y] = d3.pointer(event, svgElement);
        coordinatesStore.update((pieces) => {
            if (dragIndex !== -1) {
                pieces[dragIndex].x += x - dragStartX;
                pieces[dragIndex].y += y - dragStartY;
                dragStartX = x;
                dragStartY = y;
            }
            return pieces;
        });

        drawMarkers();
    }

    function handleMouseUp(event: MouseEvent) {
        if (!isDragging) return;

        isDragging = false;
        dragIndex = -1;
    }

    function drawMarkers() {
        const svg = d3.select(svgElement);
        svg.selectAll("text").remove();

        coordinatesStore.subscribe((pieces) => {
            svg.selectAll("text")
                .data(pieces.filter((p) => p.active))
                .enter()
                .append("text")
                .attr("x", (d) => d.x)
                .attr("y", (d) => d.y)
                .attr("class", "piece ink")
                .text((d) => d.symbol)
                .style("font-size", "3em");
        });
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
    :global(.piece) {
        font-family: Arial, sans-serif;
        transition: transform 0.3s ease-in-out;
    }
</style>
