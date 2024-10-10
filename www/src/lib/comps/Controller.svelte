<script lang="ts">
    import { onDestroy, tick } from "svelte"; // Import tick
    import { gameStore } from "$lib/store";
    import { createGame, resetGame, startGame, pauseGame, stepGame, quitGame } from "$lib/api";
    import { updateVisualization } from "$lib/plots";
    import type { State, Scenario } from "$lib/types";
    import { get } from "svelte/store";

    let history: { content: string; author: string }[] = [
        {
            content:
                "In this scenario, you are a C2 commander of the allies. Commands start with '|'. Use '|make' or '|m' followed by a place to create the game. Use '|begin' or '|b' to start the game. Use '|pause' or '|p' to pause the game, '|s' to step the game, '|reset' or '|r' to reset the state, and '|quit' or '|q' to end the game.",
            author: "bot",
        },
    ];

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
        history = [...history, { content: input.value, author: "user" }];
        input.value = "";

        if (message.startsWith("|")) {
            const [command, ...args] = message.slice(1).trim().split(" ");
            const place = args.join(" ").trim() || "Abel Cathrines Gade, Copenhagen, Denmark";
            const { gameId } = get(gameStore);

            switch (command.toLowerCase()) {
                case "make":
                case "m":
                    if (get(gameStore).gameId) await quitGame(get(gameStore).gameId!); // Use non-null assertion
                    try {
                        const { gameId, info }: { gameId: string; info: Scenario } = await createGame(place);
                        gameStore.setGame(gameId, info);
                        gameStore.setTerrain(info.terrain);

                        const { state } = await resetGame(gameId);
                        gameStore.setState(state);
                        updateVisualization();
                        history = [
                            ...history,
                            {
                                content: `Game created and reset successfully with place: ${place}. Ready to start.`,
                                author: "bot",
                            },
                        ];
                    } catch (error) {
                        console.error("Error creating or resetting game:", error);
                        history = [
                            ...history,
                            { content: "Error creating or resetting game. Please try again.", author: "bot" },
                        ];
                    }
                    break;
                case "begin":
                case "b":
                    if (gameId) {
                        try {
                            await startGame(gameId);
                            setupWebSocket(gameId);
                            history = [...history, { content: "Game simulation started.", author: "bot" }];
                        } catch (error) {
                            console.error("Error starting game simulation:", error);
                            history = [
                                ...history,
                                { content: "Error starting game simulation. Please try again.", author: "bot" },
                            ];
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
                            history = [...history, { content: "Step executed.", author: "bot" }];
                        } catch (error) {
                            console.error("Error stepping the game:", error);
                            history = [
                                ...history,
                                { content: "Error stepping the game. Please try again.", author: "bot" },
                            ];
                        }
                    }
                    break;
                case "pause":
                case "p":
                    if (gameId) {
                        try {
                            await pauseGame(gameId);
                            history = [...history, { content: "Game paused successfully.", author: "bot" }];
                        } catch (error) {
                            console.error("Error pausing the game:", error);
                            history = [
                                ...history,
                                { content: "Error pausing the game. Please try again.", author: "bot" },
                            ];
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
                            history = [...history, { content: "Game state reset successfully.", author: "bot" }];
                        } catch (error) {
                            console.error("Error resetting the game state:", error);
                            history = [
                                ...history,
                                { content: "Error resetting game state. Please try again.", author: "bot" },
                            ];
                        }
                    }
                    break;
                case "quit":
                case "q":
                    if (gameId) {
                        try {
                            await quitGame(gameId);
                            socket?.close();
                            gameStore.reset(); // This will use the empty state
                            updateVisualization(); // Update visuals to clear the grid
                            history = [
                                ...history,
                                { content: "Game simulation ended and grid reset to initial state.", author: "bot" },
                            ];
                        } catch (error) {
                            console.error("Error quitting the game:", error);
                            history = [
                                ...history,
                                { content: "Error quitting the game. Please try again.", author: "bot" },
                            ];
                        }
                    }
                    break;
                default:
                    history = [
                        ...history,
                        {
                            content:
                                "Command not recognized. Use '|make', '|begin', '|step', '|pause', '|reset', or '|quit'.",
                            author: "bot",
                        },
                    ];
            }
        } else {
            history = [
                ...history,
                { content: "Message not recognized as a command. Commands should start with '|'.", author: "bot" },
            ];
        }
    }

    function handleKeydown(event: KeyboardEvent) {
        if (event.key === "Enter") {
            send();
        }
    }

    let historyContainer: HTMLDivElement;

    async function scrollToBottom() {
        await tick(); // Wait for DOM update
        if (historyContainer) {
            historyContainer.scrollTop = historyContainer.scrollHeight;
        }
    }

    $: history, scrollToBottom();

    onDestroy(() => {
        if (socket) {
            socket.close();
        }
    });
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
        <input bind:this={input} type="text" placeholder="Type a command..." on:keydown={handleKeydown} />
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
        flex-grow: 1;
        margin-bottom: 1rem;
        /*// history should always be scrolled down to the latest message--> &*/
        scroll-behavior: smooth;
    }

    .bot {
        text-align: left;
    }

    .user {
        text-align: right;
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
