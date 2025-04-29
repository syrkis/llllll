<script lang="ts">
    import { onMount } from "svelte";
    import { init, reset, step, close } from "../lib/api";
    import { API_BASE_URL } from "../lib/utils";
    import type { LogEntry, ChatEntry, State, Scene } from "../lib/types";

    // Define types for our app state

    // Convert messages to use $state for reactivity
    let messages = $state<ChatEntry[]>([
        {
            text: "Welcome! Type '| help' for available commands.",
            user: "system",
        },
    ]);
    let gameId = $state<string | null>(null);
    let gameState = $state<State | null>(null);
    let scene = $state<Scene | null>(null);
    let logs = $state<LogEntry[]>([]);
    let error = $state<string | null>(null);
    let loading = $state(false);

    // Command history implementation
    let commandHistory = $state<string[]>([]);
    let historyIndex = $state<number>(-1);

    // Command map for aliases/shortcuts (first letter shortcuts)
    const commandAliases = {
        i: "init",
        r: "reset",
        s: "step",
        c: "close",
        h: "help",
    };

    // Log function to track API calls and results
    function addLog(message: string): void {
        logs = [...logs, { time: new Date().toLocaleTimeString(), message }];
    }

    function addMessage(message: ChatEntry): void {
        messages = [...messages, message];
    }

    // Resolve command alias to full command
    function resolveCommand(cmd: string): string {
        // If it's a single character and we have an alias for it
        if (cmd.length === 1 && cmd in commandAliases) {
            return commandAliases[cmd as keyof typeof commandAliases];
        }
        return cmd;
    }

    // Process a single command
    async function processCommand(command: string): Promise<void> {
        // Remove the leading | and trim whitespace
        const cmdRaw = command.substring(1).trim().toLowerCase();

        // Resolve any command aliases
        const cmd = resolveCommand(cmdRaw);

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
                        "| init (i) - Initialize new game\n" +
                        "| reset (r) - Reset current game\n" +
                        "| step (s) - Advance game state\n" +
                        "| close (c) - Close current game\n" +
                        "| help (h) - Show available commands\n\n" +
                        "You can use the first letter as shortcut (| i, | r, | s, | c, | h)\n" +
                        "Press Enter without command to run step\n" +
                        "Chain commands with multiple bars (| i | r | s)",
                    user: "system",
                });
                break;
            default:
                addMessage({
                    text: `Unknown command: '${cmdRaw}'. Type '| help' for available commands.`,
                    user: "system",
                });
        }
    }

    // Process multiple commands (for chaining)
    async function processCommands(commandText: string): Promise<void> {
        // Split by | and filter out empty segments
        const commands = commandText
            .split("|")
            .map((cmd) => cmd.trim())
            .filter((cmd) => cmd.length > 0)
            .map((cmd) => `|${cmd}`); // Add back the | prefix for processing

        // Process each command sequentially
        for (const cmd of commands) {
            await processCommand(cmd);
        }
    }

    // Handle form submission
    async function handleSubmit(event: SubmitEvent): Promise<void> {
        event.preventDefault();
        const messageInput = document.getElementById(
            "messageInput",
        ) as HTMLInputElement;
        const messageText = messageInput.value.trim();

        // If empty message, run step command
        if (!messageText) {
            addMessage({
                text: "| step",
                user: "person",
            });
            await stepGame();
            addMessage({ text: "Game stepped", user: "system" });
            return;
        }

        // Add the message to the chat
        addMessage({
            text: messageText,
            user: "person",
        });

        // If command, add to history
        if (messageText.startsWith("|")) {
            // Add to command history
            commandHistory = [...commandHistory, messageText];
            historyIndex = -1;
        }

        // Check if this is a command (starts with |)
        if (messageText.startsWith("|")) {
            await processCommands(messageText);
        }

        // Clear the input field
        messageInput.value = "";
    }

    // Initialize a new game
    async function initGame(): Promise<void> {
        loading = true;
        error = null;
        try {
            addLog("Initializing game for Copenhagen, Denmark...");
            const initResponse = await init("Copenhagen, Denmark");
            gameId = initResponse.game_id;
            scene = initResponse.scene;
            // console.log();
            addLog(`Game initialized with ID: ${gameId}`);
            addLog(
                `Terrain data received: ${scene.terrain ? scene.terrain.length : 0} buildings`,
            );
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
            if (gameState.unit && gameState.unit.length > 0) {
                addLog(`Units in game: ${gameState.unit.length}`);
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
            scene = null;
        } catch (err: unknown) {
            const errorMessage =
                err instanceof Error ? err.message : String(err);
            error = `Close error: ${errorMessage}`;
            addLog(`Error: ${error}`);
        } finally {
            loading = false;
        }
    }

    // Handle auto-adding space after vertical bar and command history navigation
    function handleKeydown(event: KeyboardEvent): void {
        const input = event.target as HTMLInputElement;

        // Handle up/down arrows for command history
        if (event.key === "ArrowUp") {
            if (
                commandHistory.length > 0 &&
                historyIndex < commandHistory.length - 1
            ) {
                historyIndex++;
                input.value =
                    commandHistory[commandHistory.length - 1 - historyIndex];
                // Set cursor at the end
                setTimeout(() => {
                    input.selectionStart = input.selectionEnd =
                        input.value.length;
                }, 0);
                event.preventDefault();
            }
        } else if (event.key === "ArrowDown") {
            if (historyIndex > 0) {
                historyIndex--;
                input.value =
                    commandHistory[commandHistory.length - 1 - historyIndex];
            } else if (historyIndex === 0) {
                historyIndex = -1;
                input.value = "";
            }
            event.preventDefault();
        }
    }

    // Auto-add space after vertical bar
    function handleInput(event: Event): void {
        const input = event.target as HTMLInputElement;
        const value = input.value;
        const selectionStart = input.selectionStart || 0;

        if (
            value.endsWith("|") &&
            (value.length === 1 || value[value.length - 2] !== "|")
        ) {
            // Add a space after the vertical bar
            input.value = `${value} `;
            // Position cursor after the inserted space
            setTimeout(() => {
                input.selectionStart = input.selectionEnd = selectionStart + 1;
            }, 0);
        }
    }

    // Display API Base URL for debugging
    onMount(() => {
        addLog(`API Base URL: ${API_BASE_URL}`);
        // Update help message to include shortcuts
        addMessage({
            text: "Welcome! Type '| h' for available commands and shortcuts.",
            user: "system",
        });
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
        <svg viewBox="0 0 100 100" width="100%" height="100%">
            {#if scene}
                {#each scene.terrain as row, i}
                    {#each row as col, j}
                        <rect
                            x={i - (col / 4 + 0.1) / 2}
                            y={j - (col / 4 + 0.1) / 2}
                            height={col / 4 + 0.1}
                            width={col / 4 + 0.1}
                        />
                    {/each}
                {/each}
            {/if}

            {#if gameState && scene}
                {#each gameState.unit as unit, i}
                    <!-- <circle cx={unit.x} cy={unit.y} r="1" fill="var(--blue)" /> -->
                    <circle
                        cx={unit.x}
                        cy={unit.y}
                        r="1"
                        fill={`var(--${scene.teams[i] === 0 ? "blue" : "red"})`}
                    />
                {/each}
            {/if}
        </svg>
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
                    placeholder="Type command (| i, | r, | s, | c, | h) or press Enter for step"
                    onkeydown={handleKeydown}
                    oninput={handleInput}
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
