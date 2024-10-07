<script lang="ts">
    // chat interface
    import Visualizer from "$lib/comps/Visualizer.svelte";

    import { afterUpdate } from "svelte";

    let history: { content: string; author: string }[] = [
        { content: "What place on earth do you want to simulate?", author: "bot" },
    ];

    let input: HTMLInputElement;
    function send() {
        history = [...history, { content: input.value, author: "user" }];
        input.value = "";
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

    <!-- <div class="input">
        <input bind:this={input} type="text" placeholder="Type a message..." on:keydown={handleKeydown} />
    </div> -->
</div>

<style>
    #controller {
        /*controler is 96vh high with 2 as padding above and bellow. Border is 2px solid*/
        height: 96vh;
        border: 2px solid;
        margin: 2vh 2vh 2vh 0;
    }
    .bot {
        padding: 0.5rem;
        text-align: left;
        font-family: monospace;
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
    .input {
        /* put input at the bottom of the controler / screen */
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
    }
</style>
