<script lang="ts">
    import { piecesStore } from "$lib/store";
    import { onMount } from "svelte";
    import * as d3 from "d3";

    let svgElement: SVGSVGElement;
    let isDragging = false;
    let draggedPieceIndex = -1;
    let dragStart = { x: 0, y: 0 };

    function activatePiece(index: number, x: number, y: number) {
        piecesStore.update((pieces) => {
            pieces[index].active = true;
            pieces[index].x = x;
            pieces[index].y = y;
            return pieces;
        });
    }

    function handlePieceClick(piece, index) {
        if (piece.active) {
            piecesStore.update((pieces) => {
                pieces[index].active = false; // Deactivate
                return pieces;
            });
        } else {
            activatePiece(index, 0, 0); // Logic to place the piece within SVG
        }
    }

    function drawPieces() {
        const svg = d3.select(svgElement);
        svg.selectAll("text").remove();

        svg.selectAll("text")
            .data($piecesStore)
            .enter()
            .append("text")
            .attr("x", (d) => d.x)
            .attr("y", (d) => d.y)
            .attr("class", "piece")
            .text((d) => d.symbol)
            .style("font-size", "3em")
            .style("opacity", (d) => (d.active ? 1 : 0.5))
            .on("click", handlePieceClick);
    }

    onMount(() => {
        drawPieces();
    });

    $: {
        drawPieces(); // Redraw whenever piecesStore changes
    }
</script>

<div id="pieces-container">
    {#each $piecesStore as piece, index (piece.name)}
        {#if !piece.active}
            <div
                class="inactive-piece"
                style="left: {piece.x}px; top: {piece.y}px"
                on:click={() => activatePiece(index, piece.x, piece.y)}
            >
                {piece.symbol}
            </div>
        {/if}
    {/each}
    <svg bind:this={svgElement}></svg>
</div>

<style>
    #pieces-container {
        position: absolute;
        width: 100%;
    }
    .inactive-piece {
        position: absolute;
        font-size: 3em;
        cursor: pointer;
        transition: transform 0.2s ease-in-out;
    }

    .inactive-piece:hover {
        transform: scale(1.1);
    }

    svg {
        height: 100%;
        width: 100%;
    }
</style>
