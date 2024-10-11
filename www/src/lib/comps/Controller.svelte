<script lang="ts">
    import { onDestroy, onMount, tick } from "svelte";
    import { gameStore } from "$lib/store";
    import { createGame, resetGame, startGame, pauseGame, stepGame, quitGame, sendMessage } from "$lib/api";
    import { updateVisualization } from "$lib/plots";
    import type { State, Scenario } from "$lib/types";
    import { get } from "svelte/store";

    let history: { content: string; author: string }[] = [
        {
            content:
                "In this scenario, you are a C2 commander of the allies. Commands start with '|'. Use '|init' or '|i' followed by a place to create the game. Use '|begin' or '|b' to start the game. Use '|pause' or '|p' to pause the game, '|s' to step the game, '|reset' or '|r' to reset the state, and '|quit' or '|q' to end the game.",
            author: "bot",
        },
    ];

    let commandHistory: { command: string; response: string }[] = [];
    let input: HTMLInputElement;
    let socket: WebSocket | null = null;

    function setupWebSocket(gameId: string) {
        socket = new WebSocket(`ws://localhost:8000/ws/${gameId}`);
        socket.onopen = () => {
            console.log("WebSocket connection established");
        };
        socket.onmessage = (event) => {
            try {
                const state: State = JSON.parse(event.data);
                gameStore.setState(state);
                updateVisualization();
            } catch (error) {
                console.error("Error parsing WebSocket message:", error);
            }
        };
        socket.onerror = (error) => {
            console.error("WebSocket error:", error);
        };
        socket.onclose = (event) => {
            console.log("WebSocket closed:", event);
        };
    }

    async function send() {
        const message = input.value.trim();
        input.value = ""; // Clear input field

        const { gameId } = get(gameStore);

        if (!message && gameId) {
            try {
                const state = await stepGame(gameId);
                gameStore.setState(state);
                updateVisualization();
            } catch (error) {
                console.error("Error stepping the game:", error);
                history = [...history, { content: "Error stepping the game. Please try again.", author: "bot" }];
            }
            await refocusInput();
            return;
        }

        if (message.startsWith("|")) {
            commandHistory = [...commandHistory, { command: message, response: "" }];

            const [command, ...args] = message.slice(1).trim().split(" ");
            const place = args.join(" ").trim() || "Abel Cathrines Gade, Copenhagen, Denmark";

            let commandResponse = "";

            switch (command.toLowerCase()) {
                case "init":
                case "i":
                    if (gameId) {
                        await quitGame(gameId);
                    }
                    try {
                        const { gameId, info }: { gameId: string; info: Scenario } = await createGame(place);
                        gameStore.setGame(gameId, info);
                        gameStore.setTerrain(info.terrain);

                        updateVisualization();
                        commandResponse = "initiated game";
                    } catch (error) {
                        console.error("Error creating or resetting game:", error);
                        commandResponse = "Error creating or resetting game. Please try again.";
                    }
                    break;
                case "begin":
                case "b":
                    if (gameId) {
                        try {
                            await startGame(gameId);
                            setupWebSocket(gameId);
                            commandResponse = "begun game";
                        } catch (error) {
                            console.error("Error starting game simulation:", error);
                            commandResponse = "Error starting game simulation. Please try again.";
                        }
                    }
                    break;
                case "step":
                case "s":
                    if (gameId) {
                        try {
                            const state = await stepGame(gameId);
                            gameStore.setState(state);
                            updateVisualization();
                            commandResponse = "step game";
                        } catch (error) {
                            console.error("Error stepping the game:", error);
                            commandResponse = "Error stepping the game. Please try again.";
                        }
                    }
                    break;
                case "pause":
                case "p":
                    if (gameId) {
                        try {
                            await pauseGame(gameId);
                            commandResponse = "paused game";
                        } catch (error) {
                            console.error("Error pausing the game:", error);
                            commandResponse = "Error pausing the game. Please try again.";
                        }
                    }
                    break;
                case "reset":
                case "r":
                    if (gameId) {
                        try {
                            const { state } = await resetGame(gameId);
                            gameStore.setState(state);
                            updateVisualization();
                            commandResponse = "reset state";
                        } catch (error) {
                            console.error("Error resetting the game state:", error);
                            commandResponse = "Error resetting game state. Please try again.";
                        }
                    }
                    break;
                case "quit":
                case "q":
                    if (gameId) {
                        try {
                            {
                                const emptyState: State = {
                                    unit_positions: [],
                                    unit_health: [],
                                    unit_types: [],
                                    unit_alive: [],
                                    unit_teams: [],
                                    unit_weapon_cooldowns: [],
                                    prev_movement_actions: [],
                                    prev_attack_actions: [],
                                    time: 0,
                                    terminal: false,
                                };
                                gameStore.setState(emptyState);
                                updateVisualization();
                            }

                            await quitGame(gameId);
                            socket?.close();
                            gameStore.reset();

                            commandResponse = "Game simulation ended, state cleared and grid reset to initial state.";
                        } catch (error) {
                            console.error("Error quitting the game:", error);
                            commandResponse = "Error quitting the game. Please try again.";
                        }
                    }
                    break;
                case "clear":
                case "c":
                    {
                        const emptyState: State = {
                            unit_positions: [],
                            unit_health: [],
                            unit_types: [],
                            unit_alive: [],
                            unit_teams: [],
                            unit_weapon_cooldowns: [],
                            prev_movement_actions: [],
                            prev_attack_actions: [],
                            time: 0,
                            terminal: false,
                        };
                        gameStore.setState(emptyState);
                        updateVisualization();
                        commandResponse = "clear";
                    }
                    break;
                default:
                    commandResponse = "Available commands: |init, |begin, |step, |pause, |reset, |quit, |clear";
            }

            commandHistory[commandHistory.length - 1].response = commandResponse;
        } else {
            history = [...history, { content: message, author: "user" }];
            try {
                const llmResponse = await sendMessage(message);
                history = [...history, { content: llmResponse, author: "bot" }];
            } catch (error) {
                console.error("Error processing message with LLM:", error);
                history = [...history, { content: "Error processing message. Please try again.", author: "bot" }];
            }
        }

        await refocusInput();
        scrollToBottom();
    }

    async function refocusInput() {
        await tick();
        input?.focus();
    }

    onMount(() => {
        if (typeof document !== "undefined") {
            refocusInput();
            document.addEventListener("click", handleGlobalClick);
        }
    });

    onDestroy(() => {
        if (socket) {
            socket.close();
        }
        if (typeof document !== "undefined") {
            document.removeEventListener("click", handleGlobalClick);
        }
    });

    function handleGlobalClick(event: MouseEvent) {
        if (event.target !== input) {
            refocusInput();
        }
    }

    function handleKeydown(event: KeyboardEvent) {
        if (event.key === "Enter") {
            send();
        } else if (event.key === "|") {
            setTimeout(() => {
                if (input?.value.charAt(input.value.length - 1) !== " ") {
                    input.value += " ";
                }
            }, 0);
        }
    }

    let historyContainer: HTMLDivElement;
    let commandHistoryContainer: HTMLDivElement;

    async function scrollToBottom() {
        await tick();
        if (historyContainer) {
            historyContainer.scrollTop = historyContainer.scrollHeight; // This keeps the chat history at the bottom
        }
        if (commandHistoryContainer) {
            commandHistoryContainer.scrollTop = 0; // This keeps the command history at the top
        }
    }

    $: isCommandInput = input?.value.startsWith("|");
</script>

<div id="controller">
    <div class="history" bind:this={historyContainer}>
        {#each history as message, i (message)}
            {#if message.author === "bot"}
                <div class="bot">{message.content}</div>
            {:else}
                <div class="user">{message.content}</div>
            {/if}
        {/each}
    </div>

    <div class="input">
        <input
            bind:this={input}
            type="text"
            on:keydown={handleKeydown}
            style="font-family: {isCommandInput ? 'monospace' : 'Sans-serif'}"
        />
    </div>

    <div class="command-history" bind:this={commandHistoryContainer}>
        {#each commandHistory.slice().reverse() as record}
            <div class="command">
                <strong>{record.command}</strong> — {record.response}
            </div>
        {/each}
        <div class="command-history-header">
            | Welcome to |||||| — write text without pipe to chat with the bot<br />
            | Write commands with pipe (i.e. |init [optional kwargs] to init a game)<br />
            | Valid commands: |init, |begin, |step, |pause, |reset, |quit, |clear<br />
        </div>
    </div>
</div>

<style>
    #controller {
        height: 96vh;
        margin: 2vh 2vh 2vh 0;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .history {
        overflow-y: auto;
        height: 100%;
        flex-grow: 1;
        /* margin-bottom: 10px; */
        scroll-behavior: smooth;
    }

    .command-history {
        overflow-y: auto;
        height: 7rem; /* Adjust height as needed for about two lines */
        padding: 0.5rem;
        border-radius: 5px;
        font-family: monospace;
        line-height: 1.5;
    }

    .command-history-header {
        font-weight: bold;
        margin-bottom: 0.5rem;
    }

    .bot {
        text-align: left;
    }

    .user {
        text-align: right;
    }

    .command {
        text-align: left;
    }

    .input {
        padding: 0.5rem;
    }

    .input input {
        width: calc(100% - 2rem);
        padding: 0.75rem 1rem;
        font-size: 1.2rem;
        border: 2px solid currentColor;
        background: none;
        outline: none;
        color: inherit;
        border-radius: 8px;
        transition: border-color 0.3s;
    }

    @media (max-width: 768px) {
        .bot,
        .user {
            font-size: 0.9rem;
        }

        .input input {
            font-size: 1rem;
        }
    }
</style>
