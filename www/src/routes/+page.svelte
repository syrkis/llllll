<script lang="ts">
    import { onMount } from "svelte";
    import { init, reset, step, close } from "../lib/api";
    import { API_BASE_URL } from "../lib/utils";
    import type { State } from "../lib/types";

    // Define types for our app state
    interface LogEntry {
        time: string;
        message: string;
    }

    // Game state using Svelte 5 runes
    let gameId = $state("");
    let gameState = $state<State | null>(null);
    let logs = $state<LogEntry[]>([]);
    let loading = $state(false);
    let error = $state<string | null>(null);

    // Log function to track API calls and results
    function addLog(message: string): void {
        logs = [...logs, { time: new Date().toLocaleTimeString(), message }];
    }

    // Initialize a new game
    async function initGame(): Promise<void> {
        loading = true;
        error = null;
        try {
            addLog("Initializing game for Copenhagen, Denmark...");
            gameId = await init("Copenhagen, Denmark");
            addLog(`Game initialized with ID: ${gameId}`);
        } catch (err: unknown) {
            const errorMessage =
                err instanceof Error ? err.message : String(err);
            error = `Init error: ${errorMessage}`;
            addLog(`Error: ${error}`);
        } finally {
            loading = false;
        }
    }

    // Reset the current game
    async function resetGame(): Promise<void> {
        if (!gameId) {
            addLog("No game to reset. Initialize a game first.");
            return;
        }

        loading = true;
        error = null;
        try {
            addLog(`Resetting game ${gameId}...`);
            gameState = await reset(gameId);
            addLog(`Game reset. Current step: ${gameState.step}`);
        } catch (err: unknown) {
            const errorMessage =
                err instanceof Error ? err.message : String(err);
            error = `Reset error: ${errorMessage}`;
            addLog(`Error: ${error}`);
        } finally {
            loading = false;
        }
    }

    // Advance the game state
    async function stepGame(): Promise<void> {
        if (!gameId) {
            addLog("No game to step. Initialize and reset a game first.");
            return;
        }

        loading = true;
        error = null;
        try {
            addLog(`Stepping game ${gameId}...`);
            gameState = await step(gameId);
            addLog(`Game stepped. Current step: ${gameState.step}`);

            // Log number of units
            if (gameState.units && gameState.units.length > 0) {
                addLog(`Units in game: ${gameState.units[0].length}`);
            }
        } catch (err: unknown) {
            const errorMessage =
                err instanceof Error ? err.message : String(err);
            error = `Step error: ${errorMessage}`;
            addLog(`Error: ${error}`);
        } finally {
            loading = false;
        }
    }

    // Close the current game
    async function closeGame(): Promise<void> {
        if (!gameId) {
            addLog("No game to close.");
            return;
        }

        loading = true;
        error = null;
        try {
            addLog(`Closing game ${gameId}...`);
            await close(gameId);
            addLog("Game closed successfully.");
            gameId = "";
            gameState = null;
        } catch (err: unknown) {
            const errorMessage =
                err instanceof Error ? err.message : String(err);
            error = `Close error: ${errorMessage}`;
            addLog(`Error: ${error}`);
        } finally {
            loading = false;
        }
    }

    // Display API Base URL for debugging
    onMount(() => {
        addLog(`API Base URL: ${API_BASE_URL}`);
    });
</script>

<main class="container">
    <div id="simulator"></div>
    {#if gameId}
        {#each [1, 2, 3] as index}
            <div>{gameId}</div>
        {/each}
    {/if}

    {#if error}
        <div class="error">
            <strong>Error:</strong>
            {error}
        </div>
    {/if}

    <div class="game-controls">
        <h2>Game Controls</h2>
        <div class="button-group">
            <button onclick={initGame} disabled={loading}
                >Initialize Game</button
            >
            <button onclick={resetGame} disabled={loading || !gameId}
                >Reset Game</button
            >
            <button onclick={stepGame} disabled={loading || !gameId}
                >Step Game</button
            >
            <button onclick={closeGame} disabled={loading || !gameId}
                >Close Game</button
            >
        </div>
    </div>

    {#if gameId}
        <div class="game-info">
            <h2>Game Information</h2>
            <p><strong>Game ID:</strong> {gameId}</p>

            {#if gameState}
                <div class="state-info">
                    <h3>Game State</h3>
                    <p><strong>Current Step:</strong> {gameState.step}</p>
                    <p>
                        <strong>Units Count:</strong>
                        {gameState.units?.[0]?.length || 0}
                    </p>
                </div>
            {/if}
        </div>
    {/if}

    <div class="logs">
        <h2>Logs</h2>
        <div class="log-container">
            {#each logs as log}
                <div class="log-entry">
                    <span class="log-time">[{log.time}]</span>
                    <span class="log-message">{log.message}</span>
                </div>
            {/each}
        </div>
    </div>
</main>

<style>
    .container {
        max-width: 800px;
        margin: 0 auto;
        padding: 1rem;
        font-family: sans-serif;
    }

    .error {
        /* background-color: #ffdddd; */
        /* color: #990000; */
        padding: 0.5rem;
        margin: 1rem 0;
        border-radius: 4px;
    }

    .button-group {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 1rem;
    }

    button {
        padding: 0.5rem 1rem;
        border: none;
        /* background-color: #4a4a4a; */
        /* color: white; */
        border-radius: 4px;
        cursor: pointer;
    }

    button:hover:not(:disabled) {
        background-color: #5a5a5a;
    }

    button:disabled {
        background-color: #cccccc;
        cursor: not-allowed;
    }

    .game-info,
    .game-controls {
        /* background-color: #f3f3f3; */
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }

    .log-container {
        height: 300px;
        overflow-y: auto;
        /* background-color: #888; */
        padding: 0.5rem;
        border-radius: 4px;
    }

    .log-entry {
        margin-bottom: 0.25rem;
        font-family: monospace;
        font-size: 0.85rem;
    }

    .log-time {
        color: #666;
        margin-right: 0.5rem;
    }

    .log-message {
        word-break: break-word;
    }

    .state-info {
        margin-top: 1rem;
    }
</style>
