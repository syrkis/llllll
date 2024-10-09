<slot />

<style>
    :global(.ink) {
        fill: none;
        stroke: black;
    }
    :global(.ink.enemy) {
        fill: black;
        stroke: black;
    }
    :global(body) {
        margin: 0;
        overflow: hidden;
    }

    @media (prefers-color-scheme: dark) {
        :global(.ink) {
            /* fill: white; */
            stroke: white;
        }
        :global(.ink.enemy) {
            fill: white;
            stroke: white;
        }
        :global(body) {
            background-color: black;
            color: white;
        }
    }
    :global(h1, h2) {
        margin: 0;
    }
</style>
