<slot />

<style>
    :root {
        --ally-color: #31511e;
        --enemy-color: #8b2635;
    }
    :global(.piece) {
        user-select: none;
        font-family: Arial, sans-serif;
        transition: transform 0.3s ease-in-out;
    }
    :global(.ink) {
        stroke: black;
        color: black;
    }
    :global(.ink.enemy) {
        fill: var(--enemy-color);
        stroke: var(--enemy-color);
    }
    :global(.ink.ally) {
        stroke: black;
        fill: var(--ally-color);
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
            fill: var(--enemy-color);
            stroke: var(--enemy-color);
        }
        :global(.ink.ally) {
            fill: var(--ally-color);
            stroke: var(--ally-color);
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
