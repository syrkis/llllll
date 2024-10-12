<slot />

<style>
    :global(.piece) {
        user-select: none;
        font-family: Arial, sans-serif;
        transition: transform 0.3s ease-in-out;
    }
    :global(.ink) {
        fill: black;
        stroke: black;
        color: black;
    }
    :global(.ink.enemy) {
        fill: black;
        stroke: black;
    }
    :global(.ink.ally) {
        fill: white;
        stroke: black;
    }
    :global(body) {
        margin: 0;
        overflow: hidden;
        font-family: monospace;
        font-size: 1.1rem;
        line-height: 2;
    }

    @media (prefers-color-scheme: dark) {
        :global(.ink) {
            fill: white;
            stroke: white;
            color: white;
        }
        :global(.ink.enemy) {
            fill: white;
            stroke: white;
        }
        :global(.ink.ally) {
            fill: black;
            stroke: white;
        }
        :global(body) {
            background-color: black;
            color: white;
            user-select: none; /* Standard syntax */
        }
    }
    :global(h1, h2) {
        margin: 0;
    }
</style>
