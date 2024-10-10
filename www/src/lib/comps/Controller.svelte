<script lang="ts">
    import { onDestroy } from "svelte";
    import { gameStore } from "$lib/store";
    import { createGame, resetGame, startGame } from "$lib/api";
    import { updateVisualization } from "$lib/plots";
    import type { State, Scenario } from "$lib/types";
    import { get } from "svelte/store";

    let history: { content: string; author: string }[] = [
        {
            content:
                "In this scenario, you are a C2 commander of the allies. Commands start with '|'. Use '|make' or '|m' followed by a place to create the game. Use '|simulate' or '|s' to start the game.",
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
            // Parse command by splitting the message
            const [command, ...args] = message.slice(1).trim().split(" ");
            const place =
                args.join(" ").trim() || get(gameStore).gameInfo?.place || "Abel Cathrines Gade, Copenhagen, Denmark";

            // Implement command logic
            if (["make", "m"].includes(command.toLowerCase())) {
                try {
                    const { gameId, info }: { gameId: string; info: Scenario } = await createGame(place);
                    gameStore.setGame(gameId, info);
                    gameStore.setTerrain(info.terrain);

                    console.log("Terrain updated:", info.terrain);
                    updateVisualization();
                    history = [
                        ...history,
                        { content: `Game created successfully with place: ${place}`, author: "bot" },
                    ];
                } catch (error) {
                    console.error("Error creating game:", error);
                    history = [...history, { content: "Error creating game. Please try again.", author: "bot" }];
                }
            } else if (["simulate", "s"].includes(command.toLowerCase())) {
                const { gameId } = get(gameStore);
                if (gameId) {
                    try {
                        await resetGame(gameId);
                        await startGame(gameId);
                        setupWebSocket(gameId);
                        history = [...history, { content: "Game started successfully.", author: "bot" }];
                    } catch (error) {
                        console.error("Error starting game:", error);
                        history = [...history, { content: "Error starting game. Please try again.", author: "bot" }];
                    }
                } else {
                    history = [
                        ...history,
                        { content: "No game has been made. Please create a game with '|make' first.", author: "bot" },
                    ];
                }
            } else {
                history = [
                    ...history,
                    {
                        content: "Command not recognized. Use '|make' to create and '|simulate' to begin the game.",
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
        font-size: 1rem;
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
        border: 2px solid currentColor; /* Uses the current text color */
        background: none; /* No background */
        outline: none; /* Removes default input outline */
        color: inherit; /* Inherits the text color */
        border-radius: 8px; /* Rounded corners for a softer look */
        transition: border-color 0.3s; /* Smooth transition for border color */
    }

    @media (max-width: 768px) {
        /* Responsive adjustments for smaller screens */
        .bot,
        .user {
            font-size: 0.9rem;
        }

        .input input {
            font-size: 1rem;
        }
    }
</style>
