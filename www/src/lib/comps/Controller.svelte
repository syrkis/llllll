<script lang="ts">
    import Visualizer from "$lib/comps/Visualizer.svelte";
    import { afterUpdate } from "svelte";
    import { createGame, resetGame, startGame } from "$lib/api";
    import { gameStore } from "$lib/store";

    let history: { content: string; author: string }[] = [
        {
            content:
                "In this scenario, the war is global, and yet fought with guns. To do your part, you must act as a C2 commander of the allies. Type 'start' to begin the game.",
            author: "bot",
        },
    ];

    let input: HTMLInputElement;
    let gameId: string | null = null;

    async function send() {
        const message = input.value.trim().toLowerCase();
        history = [...history, { content: input.value, author: "user" }];
        input.value = "";

        if (message === "start") {
            try {
                if (!gameId) {
                    const { gameId: newGameId, info } = await createGame("Marmorkirken, Copenhagen, Denmark");
                    gameId = newGameId;
                    gameStore.setGame(newGameId, info);
                    await resetGame(newGameId);
                }
                await startGame(gameId);
                history = [...history, { content: "Game started successfully!", author: "bot" }];
            } catch (error) {
                console.error("Error starting game:", error);
                history = [...history, { content: "Error starting game. Please try again.", author: "bot" }];
            }
        } else {
            // Handle other commands or messages here
            history = [
                ...history,
                { content: "Command not recognized. Type 'start' to begin the game.", author: "bot" },
            ];
        }
    }

    function handleKeydown(event: KeyboardEvent) {
        if (event.key === "Enter") {
            send();
        }
    }

    let historyContainer: HTMLDivElement;

    afterUpdate(() => {
        if (historyContainer) {
            const lastMessage = historyContainer.lastElementChild;
            if (lastMessage) {
                lastMessage.scrollIntoView({ behavior: "smooth" });
            }
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
        <input bind:this={input} type="text" placeholder="Type a message..." on:keydown={handleKeydown} />
    </div>
</div>

<style>
    #controller {
        /*controler is 96vh high with 2 as padding above and bellow. Border is 2px solid*/
        height: 96vh;
        /* border: 2px solid; */
        margin: 2vh 2vh 2vh 0;
    }
    .bot {
        padding: 0.5rem;
        text-align: left;
        letter-spacing: 0.08rem;
        line-height: 3.5rem;
        text-align: justify;
        font-size: 2rem;
        font-family: "IBM Plex Sans", sans-serif;
    }
    .user {
        padding: 0.5rem;
        text-align: right;
    }

    /* enable scroll through history */
    .history {
        overflow-y: auto;
        height: calc(50% - 3rem);
    }

    /* visualizer is the top  half of the controler / screen  (50% height) */
    /* .input {
        put input at the bottom of the controler / screen
        position: absolute;
        bottom: 0;
        display: flex;
        width: calc(100vw - 100vh);
    }
    input {
        width: 100%;
        height: 2rem;
        font-size: 1rem;
        padding: 0.5rem;
        background-color: black;
        color: white;
        border-radius: 0.5rem;
        border: solid 4px rgba(0, 0, 0, 0.1);
    } */
</style>
