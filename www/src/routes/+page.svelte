<script lang="ts">
    import { onMount } from "svelte";
    import { init, reset, step, close } from "../lib/api";
    import { API_BASE_URL, resolveCommand } from "../lib/utils";
    import type {
        LogEntry,
        ChatEntry,
        State,
        Scene,
        Pieces,
    } from "../lib/types";

    // Convert messages to use $state for reactivity
    let messages = $state<ChatEntry[]>([{ text: "Welcome!", user: "system" }]);
    let pieces = $state<Pieces>({
        king: null,
        queen: null,
        rook: null,
        bishop: null,
        knight: null,
        pawn: null,
    });
    let gameId = $state<string | null>(null);
    let gameState = $state<State | null>(null);
    let scene = $state<Scene>({
        terrain: [...Array(100)].map(() => Array(100).fill(0)),
    });
    let logs = $state<LogEntry[]>([]);
    let error = $state<string | null>(null);
    let loading = $state(false);

    // Command history implementation
    let commandHistory = $state<string[]>([]);
    let historyIndex = $state<number>(-1);

    // Log function to track API calls and results
    function addLog(message: string): void {
        logs = [...logs, { time: new Date().toLocaleTimeString(), message }];
    }

    function addMessage(message: ChatEntry): void {
        messages = [...messages, message];
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
                // Log to logs but don't add to messages
                addLog("Game initialized");
                break;
            case "reset":
                await resetGame();
                addLog("Game reset");
                break;
            case "step":
                await stepGame();
                addLog("Game stepped");
                break;
            case "close":
                await closeGame();
                addLog("Game closed");
                break;
            case "help":
                addLog(
                    "Available commands:\n" +
                        "| init (i) - Initialize new game\n" +
                        "| reset (r) - Reset current game\n" +
                        "| step (s) - Advance game state\n" +
                        "| close (c) - Close current game\n" +
                        "| help (h) - Show available commands\n\n" +
                        "You can use the first letter as shortcut (| i, | r, | s, | c, | h)\n" +
                        "Press Enter without command to run step\n" +
                        "Chain commands with multiple bars (| i | r | s)",
                );
                break;
            default:
                // Log unknown command to logs, not messages
                addLog(
                    `Unknown command: '${cmdRaw}'. Type '| help' for available commands.`,
                );
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

        // If empty message, run step command silently
        if (!messageText) {
            await stepGame();
            return;
        }

        // If command (starts with |), add to history but don't show in chat
        if (messageText.startsWith("|")) {
            // Add to command history
            commandHistory = [...commandHistory, messageText];
            historyIndex = -1;

            // Process the command(s)
            await processCommands(messageText);
        } else {
            // If it's not a command, add it to the chat as a regular message
            addMessage({
                text: messageText,
                user: "person",
            });
        }

        // Clear the input field
        messageInput.value = "";

        // Remove command mode class when input is cleared
        messageInput.classList.remove("command-mode");
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
            // Add more detailed logs instead of system messages
            addLog(
                `Terrain loaded for Copenhagen, Denmark. Use '| reset' to prepare the game.`,
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
            addLog("No game to reset. Initialize a game first with '| init'.");
            return;
        }

        loading = true;
        error = null;
        try {
            addLog(`Resetting game ${gameId}...`);
            gameState = await reset(gameId, scene);
            addLog(`Game reset. Current step: ${gameState.step}`);

            // Add more detailed logs instead of system messages
            addLog(
                `Units placed on the map. Use '| step' or press Enter to advance the simulation.`,
            );
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
            addLog(
                "No game to step. Initialize and reset a game first with '| init'.",
            );
            return;
        }

        loading = true;
        error = null;
        try {
            addLog(`Stepping game ${gameId}...`);
            gameState = await step(gameId, scene);
            addLog(`Game stepped. Current step: ${gameState.step}`);

            // Log number of units
            let unitCount = 0;
            if (gameState.unit && gameState.unit.length > 0) {
                unitCount = gameState.unit.length;
                addLog(`Units in game: ${unitCount}`);
            }

            addLog(
                `Step ${gameState.step} completed. ${unitCount} units active.`,
            );
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
            const currentGameId = gameId;
            addLog(`Closing game ${currentGameId}...`);
            await close(currentGameId);
            addLog("Game closed successfully.");
            gameId = null;
            gameState = null;
            scene = {
                terrain: Array(100)
                    .fill(0)
                    .map(() => Array(100).fill(0)),
            };

            // Add informative log instead of system message
            addLog(
                `Game ${currentGameId.substring(0, 8)}... closed successfully. Use '| init' to start a new game.`,
            );
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

                // Update command mode class
                if (input.value.trim().startsWith("|")) {
                    input.classList.add("command-mode");
                } else {
                    input.classList.remove("command-mode");
                }
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

            // Update command mode class
            if (input.value.trim().startsWith("|")) {
                input.classList.add("command-mode");
            } else {
                input.classList.remove("command-mode");
            }
        }
    }

    // Auto-add spaces before and after vertical bar
    function handleInput(event: Event): void {
        const input = event.target as HTMLInputElement;
        const value = input.value;
        const selectionStart = input.selectionStart || 0;

        // Toggle command-mode class based on if input starts with |
        if (value.trim().startsWith("|")) {
            input.classList.add("command-mode");
        } else {
            input.classList.remove("command-mode");
        }

        // Only process if the bar character was just typed (cursor position right after a bar)
        const barJustTyped =
            selectionStart > 0 && value.charAt(selectionStart - 1) === "|";

        if (barJustTyped) {
            const originalPosition = selectionStart;
            let newValue = value;
            let finalCursorPosition: number;

            // Check if we need to add space before the bar (not at start and no space before)
            const addSpaceBefore =
                originalPosition > 1 &&
                value.charAt(originalPosition - 2) !== " ";

            // Always add space after bar
            newValue = `${value.substring(0, originalPosition)} ${value.substring(originalPosition)}`;

            // Add space before if needed
            if (addSpaceBefore) {
                // Calculate position of bar after adding the space after
                const barPosAfterFirstMod = originalPosition;

                // Insert space before the bar
                newValue = `${newValue.substring(0, barPosAfterFirstMod - 1)} ${newValue.substring(barPosAfterFirstMod - 1)}`;

                // Final cursor position will be after the bar and the space we added
                finalCursorPosition = originalPosition + 2; // +1 for space before, +1 for space after
            } else {
                // No space before needed, cursor will be after bar and the space
                finalCursorPosition = originalPosition + 1; // +1 for space after
            }

            // Update the input value
            input.value = newValue;

            // Position cursor correctly
            setTimeout(() => {
                input.selectionStart = input.selectionEnd = finalCursorPosition;
            }, 0);
        }
    }
    // function grow(node) {
    //     if (node.getAttribute("r") === "0") {
    //         setTimeout(() => node.setAttribute("r", "1"), 10);
    //     }
    //     return {};
    // }

    // Display API Base URL for debugging
    onMount(() => {
        addLog(`API Base URL: ${API_BASE_URL}`);

        // Initialize command mode for input field if needed
        const messageInput = document.getElementById(
            "messageInput",
        ) as HTMLInputElement;
        if (messageInput?.value.trim().startsWith("|")) {
            messageInput.classList.add("command-mode");
        }
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
        <svg
            viewBox="0 0 100 100"
            width="100%"
            height="100%"
            preserveAspectRatio="xMidYMid meet"
        >
            {#each scene.terrain as row, i}
                {#each row as col, j}
                    <rect
                        x={i - (col / 2 + 0.1) / 2 + 0.5}
                        y={j - (col / 2 + 0.1) / 2 + 0.5}
                        height={col / 2 + 0.1}
                        width={col / 2 + 0.1}
                    />
                {/each}
            {/each}

            {#if gameState && scene.cfg}
                {#each gameState.unit as unit, i (unit.id || i)}
                    <circle
                        cx={unit.x}
                        cy={unit.y}
                        r={gameState.step <= 1 ? "0" : "1"}
                        fill={`var(--${scene.cfg.teams[i] === 0 ? "red" : "blue"})`}
                    />
                {/each}
            {/if}
        </svg>
    </div>

    <div id="controler">
        <!-- chat history - will grow/shrink based on available space -->
        <div class="chat-history">
            {#each messages as message}
                {#if message.user === "system"}
                    <div class="system">{message.text}</div>
                {:else}
                    <div class="person">{message.text}</div>
                {/if}
            {/each}
        </div>

        <!-- Fixed elements at the bottom -->
        <div class="bottom-section">
            <!-- chat input -->
            <form onsubmit={handleSubmit} autocomplete="off">
                <div class="input-container">
                    <input
                        type="text"
                        class="input-container command-mode"
                        id="messageInput"
                        placeholder="Type command (| i, | r, | s, | c, | h) or press Enter for step"
                        onkeydown={handleKeydown}
                        oninput={handleInput}
                        autocomplete="off"
                    />
                </div>
            </form>

            <!-- log history -->
            <div class="log-history">
                {#each [...logs].reverse() as log}
                    <div class="log-entry">
                        <span class="log-time">[{log.time}]</span>
                        <span class="log-message">{log.message}</span>
                    </div>
                {/each}
            </div>
        </div>
    </div>
</main>
