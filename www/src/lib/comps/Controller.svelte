<script lang="ts">
    // chat interface
    import Visualizer from "$lib/comps/Visualizer.svelte";

    import { afterUpdate } from "svelte";

    let history: { content: string; author: string }[] = [
        {
            content: `Welcome to the life like command and control (C2) simulator. You are in a C2 center, far removed from the war.`,
            author: "bot",
        },
        {
            content: `In this scenario, the war is global, and yet fought with guns. To do your part, you must act as a C2 commander of the allies.`,
            author: "bot",
        },
        {
            content: `You are fighting the enemies. Your troops can hide and move through trees, shoot but not cross water, while buildings are impenetrable.`,
            author: "bot",
        },
        {
            content: `Tell me what place on earth you would like to command. Kongens Have, Copenhagen, Denmark, could use your help.`,
            author: "bot",
        },
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
