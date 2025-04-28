<script lang="ts">
    import { onMount } from "svelte";
    import { init, reset, step, close } from "../lib/api";
    import { API_BASE_URL } from "../lib/utils";
    import type { State, Unit } from "../lib/types";

    // Define types for our app state
    interface LogEntry {
        time: string;
        message: string;
    }

    interface Message {
        text: string;
        user: "person" | "system";
    }

    // Convert messages to use $state for reactivity
    let messages = $state<Message[]>([
        {
            text: "Welcome! Type '| help' for available commands.",
            user: "system",
        },
    ]);
    let gameId = $state<string | null>(null);
    let gameState = $state<State | null>(null);
    let logs = $state<LogEntry[]>([]);
    let error = $state<string | null>(null);
    let loading = $state(false);

    // Log function to track API calls and results
    function addLog(message: string): void {
        logs = [...logs, { time: new Date().toLocaleTimeString(), message }];
    }

    function addMessage(message: Message): void {
        messages = [...messages, message];
    }

    // Handles command processing for commands starting with |
    async function processCommand(command: string): Promise<void> {
        // Remove the leading | and trim whitespace
        const cmd = command.substring(1).trim().toLowerCase();

        switch (cmd) {
            case "init":
                await initGame();
                addMessage({ text: "Game initialized", user: "system" });
                break;
            case "reset":
                await resetGame();
                addMessage({ text: "Game reset", user: "system" });
                break;
            case "step":
                await stepGame();
                addMessage({ text: "Game stepped", user: "system" });
                break;
            case "close":
                await closeGame();
                addMessage({ text: "Game closed", user: "system" });
                break;
            case "help":
                addMessage({
                    text:
                        "Available commands:\n" +
                        "| init - Initialize new game\n" +
                        "| reset - Reset current game\n" +
                        "| step - Advance game state\n" +
                        "| close - Close current game\n" +
                        "| help - Show available commands",
                    user: "system",
                });
                break;
            default:
                addMessage({
                    text: `Unknown command: '${cmd}'. Type '| help' for available commands.`,
                    user: "system",
                });
        }
    }

    // Handle form submission
    async function handleSubmit(event: SubmitEvent): void {
        event.preventDefault();
        const messageInput = document.getElementById(
            "messageInput",
        ) as HTMLInputElement;
        const messageText = messageInput.value.trim();

        if (messageText) {
            // Add the message to the chat
            addMessage({
                text: messageText,
                user: "person",
            });

            // Check if this is a command (starts with |)
            if (messageText.startsWith("|")) {
                await processCommand(messageText);
            }

            // Clear the input field
            messageInput.value = "";
        }
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
            addMessage({
                text: "No game to reset. Initialize a game first with '| init'.",
                user: "system",
            });
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
            addMessage({
                text: "No game to step. Initialize a game first with '| init'.",
                user: "system",
            });
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
                addLog(`Units in game: ${gameState.units.length}`);
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
            addMessage({ text: "No game to close.", user: "system" });
            return;
        }

        loading = true;
        error = null;
        try {
            addLog(`Closing game ${gameId}...`);
            await close(gameId);
            addLog("Game closed successfully.");
            gameId = null;
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
    {#if error}
        <div class="error">
            <strong>Error:</strong>
            {error}
        </div>
    {/if}

    <div id="simulator">
        {#if gameState != null}
            {$inspect(gameState)}
            {#each gameState.units as unit}
                <div>{unit.x}, {unit.y}</div>
            {/each}
        {/if}
    </div>

    <div id="controler">
        <!-- chat history -->
        {#each messages as message}
            {#if message.user === "system"}
                <div class="system">{message.text}</div>
            {:else}
                <div class="person">{message.text}</div>
            {/if}
        {/each}

        <!-- chat input -->
        <form onsubmit={handleSubmit}>
            <div class="input-container">
                <input
                    type="text"
                    id="messageInput"
                    placeholder="Type message or command (| init, | reset, | step, | close)"
                    required
                />
            </div>
        </form>

        <!-- log history -->
        <div class="log-history">
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
    .system,
    .person {
        padding: 0.5rem;
        margin-bottom: 0.5rem;
        border-radius: 0.25rem;
        max-width: 80%;
        white-space: pre-wrap;
    }

    .system {
        background-color: #f0f0f0;
        align-self: flex-start;
    }

    .person {
        background-color: #e0f0ff;
        align-self: flex-end;
        margin-left: auto;
    }

    /* Container for messages */
    #controler {
        display: flex;
        flex-direction: column;
        /* keep your existing styles for #controler here */
    }

    /* Optional: Add some style to the input container */
    .input-container {
        display: flex;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .input-container input {
        flex-grow: 1;
        padding: 0.5rem;
        border-radius: 4px 0 0 4px;
        border: 1px solid #ccc;
    }

    .container {
        width: 100%;
        height: 100vh; /* Full viewport height */
        display: flex; /* Create a flex container for columns */
        font-family: sans-serif;
        overflow: hidden; /* Prevent scrolling */
    }

    #simulator {
        height: 100%; /* Full height */
        aspect-ratio: 1/1; /* Keep it square */
        border: 1px solid; /* Add a border */
        /* background-color: #f0f0f0; /* Optional: just to visualize */
    }

    #controler {
        flex: 1; /* Take up remaining space */
        height: 100%; /* Full height */
        overflow-y: auto; /* Allow scrolling if content overflows */
        padding: 10px; /* Optional: add some padding */
        border: 1px solid; /* Add a border */
        /* background-color: #e0e0e0; /* Optional: just to visualize */
    }

    .error {
        /* background-color: #ffdddd; */
        /* color: #990000; */
        padding: 0.5rem;
        margin: 1rem 0;
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
</style>
