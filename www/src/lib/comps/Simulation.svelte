<script lang="ts">
    import { onMount, onDestroy } from "svelte";
    import { gameStore, scale, piecesStore } from "$lib/store";
    import { createBackgroundGrid } from "$lib/scene";
    import { get } from "svelte/store";
    import * as d3 from "d3";

    let svgElement: SVGSVGElement;
    let initialRenderedTerrain: number[][] | null = null;

    let isDragging = false;
    let dragIndex = -1;
    let dragStartX = 0;
    let dragStartY = 0;

    const threshold = 20; // Tolerance for selecting a piece

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
        }
    }

    function handleSVGClick(event: MouseEvent) {
        if (isDragging) return;

        event.preventDefault();
        const [x, y] = d3.pointer(event, svgElement);

        piecesStore.update((pieces) => {
            let pieceIndex = pieces.findIndex(
                (p) => p.active && Math.abs(p.x - x) <= threshold && Math.abs(p.y - y) <= threshold,
            );

            if (pieceIndex !== -1) {
                pieces[pieceIndex].active = false;
            } else {
                pieceIndex = pieces.findIndex((p) => !p.active);
                if (pieceIndex !== -1) {
                    pieces[pieceIndex].x = x;
                    pieces[pieceIndex].y = y;
                    pieces[pieceIndex].active = true;
                }
            }
            return pieces;
        });

        drawMarkers(); // Force redraw
    }

    function handleMouseDown(event: MouseEvent) {
        const [x, y] = d3.pointer(event, svgElement);
        isDragging = false;

        piecesStore.update((pieces) => {
            dragIndex = pieces.findIndex(
                (p) => p.active && Math.abs(p.x - x) <= threshold && Math.abs(p.y - y) <= threshold,
            );

            if (dragIndex !== -1) {
                dragStartX = x;
                dragStartY = y;
            }
            return pieces;
        });
    }

    function handleMouseMove(event: MouseEvent) {
        if (dragIndex === -1) return;

        const [x, y] = d3.pointer(event, svgElement);
        if (Math.abs(x - dragStartX) > 0 || Math.abs(y - dragStartY) > 0) {
            isDragging = true;
        }

        if (isDragging) {
            piecesStore.update((pieces) => {
                if (dragIndex !== -1) {
                    pieces[dragIndex].x += x - dragStartX;
                    pieces[dragIndex].y += y - dragStartY;
                    dragStartX = x;
                    dragStartY = y;
                }
                return pieces;
            });

            drawMarkers(); // Continuously draw markers during drag
        }
    }

    function handleMouseUp(event: MouseEvent) {
        dragIndex = -1;
    }

    type Piece = {
        symbol: string;
        x: number;
        y: number;
        active: boolean;
    };

    // Ensure that drawMarkers runs whenever the piecesStore updates
    $: drawMarkers();

    function drawMarkers() {
        const svg = d3.select<SVGSVGElement, unknown>(svgElement);

        const activePieces = $piecesStore.filter((p) => p.active);
        const pieces = svg.selectAll<SVGTextElement, Piece>("text").data(activePieces, (d) => d.symbol);

        // Handle exiting pieces with transition
        pieces.exit().transition().duration(300).style("opacity", 0).style("font-size", "0em").remove();

        // Enter active pieces
        const piecesEnter = pieces
            .enter()
            .append("text")
            .attr("x", (d) => d.x)
            .attr("y", (d) => d.y)
            .attr("class", "piece ink")
            .attr("text-anchor", "middle")
            .attr("dominant-baseline", "central")
            .text((d) => d.symbol)
            .style("font-size", "0em")
            .style("opacity", 0);

        piecesEnter.transition().duration(500).style("font-size", "3em").style("opacity", 1);

        // Update active elements positioning
        pieces
            .attr("x", (d) => d.x)
            .attr("y", (d) => d.y)
            .style("font-size", "3em")
            .style("opacity", 1);
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
        }
    }
</script>

<div id="simulation">
    <svg bind:this={svgElement}></svg>
</div>

<style>
    #simulation {
        padding: min(2vh, 2vw);
        height: min(96vh, 96vw);
        width: min(96vh, 96vw);
    }

    svg {
        height: 100%;
        width: 100%;
    }
</style>
