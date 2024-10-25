<script lang="ts">
    import { onDestroy, onMount, tick } from "svelte";
    import { gameStore, piecesStore } from "$lib/store";
    import { createGame, resetGame, startGame, pauseGame, stepGame, quitGame, sendMessage } from "$lib/api";
    import { updateVisualization } from "$lib/plots";
    import type { State, Scenario, ChessPiece } from "$lib/types";
    import { get } from "svelte/store";

    let history: {
        content: string;
        author: string;
        type: "command" | "chat";
        response?: string;
    }[] = [
        {
            content:
                "This is a wargame. You interact with it by sending messages to me, your loving AI co-commander, or by typing commands starting with a pipe (e.g., | init Copenhagen, Denmark to run a simulation there). You can also click on the map to place and move pins to reference locations. Unlike traditional war games, you do not assign actions to units. Instead, you talk with me, and I will relay your orders and plans. For example, you could say, 'Send snipers towards the northern bridge and attack on sight, while soldiers sneak through the forest.'",
            author: "bot",
            type: "chat",
        },
    ];

    let input: HTMLInputElement;
    let socket: WebSocket | null = null;
    let historyIndex = -1;

    let pieces: ChessPiece[] = [];
    piecesStore.subscribe((value) => {
        pieces = value;
    });

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
        input.value = "";
        historyIndex = -1;

        const { gameId } = get(gameStore);

        if (!message && gameId) {
            try {
                const state = await stepGame(gameId);
                gameStore.setState(state);
                updateVisualization();
            } catch (error) {
                console.error("Error stepping the game:", error);
                history = [
                    ...history,
                    {
                        content: "Error stepping the game. Please try again.",
                        author: "bot",
                        type: "chat",
                    },
                ];
            }
            await refocusInput();
            return;
        }

        if (message.startsWith("|")) {
            const commandEntry = { content: message, author: "user", type: "command" };
            history = [...history, commandEntry];
            await processCommand(commandEntry, message, gameId);
            history = [...history];
        } else {
            history = [...history, { content: message, author: "user", type: "chat" }];
            try {
                const llmResponse = await sendMessage(message);
                history = [...history, { content: llmResponse, author: "bot", type: "chat" }];
            } catch (error) {
                console.error("Error processing message with LLM:", error);
                history = [
                    ...history,
                    { content: "Error processing message. Please try again.", author: "bot", type: "chat" },
                ];
            }
        }

        await refocusInput();
        scrollToBottom();
    }

    async function processCommand(commandEntry: any, message: string, gameId: string) {
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
                        // Set pieces to inactive
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
                        // piecesStore.update((pieces) => {
                        //     return pieces.map((piece) => ({ ...piece, active: false }));
                        // });
                    } catch (error) {
                        console.error("Error quitting the game:", error);
                        commandResponse = "Error quitting the game. Please try again.";
                    }
                }
                break;

            default:
                commandResponse = "Available commands: |init, |begin, |step, |pause, |reset, |quit, |clear";
        }

        commandEntry.response = commandResponse;
        history = [...history];
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
        } else if (event.key === "ArrowUp") {
            navigateHistory(1);
        } else if (event.key === "ArrowDown") {
            navigateHistory(-1);
        }
    }

    function navigateHistory(direction: number) {
        const userEntries = history.filter((entry) => entry.author === "user");
        if (historyIndex + direction >= 0 && historyIndex + direction < userEntries.length) {
            historyIndex += direction;
            input.value = userEntries[userEntries.length - 1 - historyIndex].content;
        } else if (historyIndex + direction < 0) {
            historyIndex = -1;
            input.value = "";
        }
    }

    let historyContainer: HTMLDivElement;
    let commandHistoryContainer: HTMLDivElement;

    async function scrollToBottom() {
        await tick();
        if (historyContainer) {
            historyContainer.scrollTop = historyContainer.scrollHeight;
        }
        if (commandHistoryContainer) {
            commandHistoryContainer.scrollTop = 0;
        }
    }

    $: isCommandInput = input?.value.startsWith("|");
</script>

<div id="controller">
    <div class="history" bind:this={historyContainer}>
        {#each history as message, i (message)}
            {#if message.type === "chat"}
                {#if message.author === "bot"}
                    <div class="bot">{message.content}</div>
                {:else}
                    <div class="user">{message.content}</div>
                {/if}
            {/if}
        {/each}
    </div>

    <div class="pieces">
        <ul>
            {#each pieces as piece}
                <li id={piece.name.toLowerCase()} class:inactive={piece.active}>
                    {piece.symbol}
                </li>
            {/each}
        </ul>
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
        {#each history.filter((item) => item.type === "command").reverse() as record (record)}
            <div class="command">
                <strong>{record.content}</strong>
                {#if record.response}
                    — {record.response}{/if}
            </div>
        {/each}
        <div class="command-history-header">
            ||||||| — a language based command and control simulator
            <br />
            | Run the commands with prefix | to control the game
            <br />
            | Commands: |init, |begin, |step, |pause, |reset, |quit<br />
        </div>
    </div>
</div>

<style>
    #controller {
        display: flex;
        flex-direction: column;
        height: 96vh; /* Default to using most of the viewport height */
        margin: 2vh;
    }

    @media (max-aspect-ratio: 1/1) {
        #controller {
            height: 96vw;
        }
    }

    li {
        transition:
            font-size 0.5s ease,
            opacity 0.5s ease;
    }

    .pieces ul {
        list-style-type: none;
        display: flex;
        justify-content: space-between;
        padding: 0;
        margin: 0;
    }

    .pieces li {
        display: inline-block;
        font-size: 3rem;
        opacity: 1;
        transition:
            font-size 0.5s ease,
            opacity 0.5s ease;
        text-align: center;
        width: 3rem; /* Maintain consistent space */
        height: 3rem; /* Consistent height */
    }

    .pieces li.inactive {
        font-size: 0rem;
        opacity: 0;
        transition:
            font-size 0.5s ease,
            opacity 0.5s ease 0.5s; /* Ensure transitions take place sequentially */
    }
    .pieces {
        padding: 40px 1rem;
        text-align: center;
    }

    .history {
        flex: 1; /* Take up all available space */
        overflow-y: auto;
        scroll-behavior: smooth;
    }

    .command-history {
        height: 120px;
        overflow-y: auto;
        padding: 0.5rem;
        border-radius: 5px;
        font-family: monospace;
    }

    .command-history-header {
        font-size: inherit;
        font-weight: bold;
        font-size: 0.9rem;
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
        padding-top: 1rem;
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
        .command-history,
        .command-history-header {
            font-size: 0.7rem;
        }

        .input input {
            font-size: 1rem;
        }
    }
</style>
