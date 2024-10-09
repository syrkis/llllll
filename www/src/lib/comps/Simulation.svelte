<script lang="ts">
    import { onMount, onDestroy } from "svelte";
    import { gameStore, scale } from "$lib/store";
    import { get } from "svelte/store";
    import { createGame, resetGame, startGame } from "$lib/api";
    import { createBackgroundGrid } from "$lib/scene";
    import { updateVisualization } from "$lib/plots";
    import * as d3 from "d3";
    import type { Observation, State } from "$lib/types";

    const place = "Marmorkirken, Copenhagen, Denmark";

    let isMounted = false;
    let svgElement: SVGSVGElement;
    let socket: WebSocket;

    function setupWebSocket(gameId: string) {
        socket = new WebSocket(`ws://localhost:8000/ws/${gameId}`);
        socket.onopen = () => {
            console.log("WebSocket connection established");
        };
        socket.onmessage = (event) => {
            console.log("Received WebSocket message:", event.data);
            const state: State = JSON.parse(event.data);
            console.log("Parsed state:", state);
            console.log("New unit positions:", state.unit_positions);
            gameStore.setState(state);
            console.log("Updated game store state:", get(gameStore).currentState);
            updateVisualization();
        };
        // ... (keep other WebSocket handlers)
    }
    function initializeScale() {
        if (isMounted && svgElement) {
            const width = svgElement.clientWidth;
            const height = svgElement.clientHeight;
            const newScale = d3
                .scaleLinear()
                .domain([0, 100])
                .range([0, Math.min(width, height)]);
            scale.set(newScale);
        }
    }

    async function initializeGame() {
        try {
            const { gameId, info } = await createGame(place);
            gameStore.setGame(gameId, info);
            console.log("Game created with ID:", gameId);
            console.log("Game info:", info);

            initializeScale();

            const { obs, state }: { obs: Observation; state: State } = await resetGame(gameId);
            gameStore.setState(state);
            console.log("Initial game state:", state);
            console.log("Initial observation:", obs);

            // Set up WebSocket connection
            setupWebSocket(gameId);

            // Render the initial state
            updateVisualization();
        } catch (error) {
            console.error("Error initializing or resetting game:", error);
        }
    }

    onMount(() => {
        isMounted = true;
        initializeGame();
        if (typeof window !== "undefined") {
            window.addEventListener("resize", initializeScale);
        }
    });

    onDestroy(() => {
        if (typeof window !== "undefined") {
            window.removeEventListener("resize", initializeScale);
        }
        if (socket) {
            socket.close();
        }
    });

    $: if ($gameStore.gameInfo && $scale) {
        console.log("Game store or scale updated, calling updateVisualization");
        updateVisualization();
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
