<slot />

<style>
    :root {
        --ally-color: #4a7c2a; /* Military green, easier to see than original */
        --enemy-color: #bc2f41; /* Deeper red, more visible but not too aggressive */
        --trees-color: #2d8c4b; /* Darker, more natural forest green */
        --water-color: #2e4899; /* Slightly darker blue, less saturated */
    }
    :global(.piece) {
        user-select: none;
        font-family: Arial, sans-serif;
        transition: transform 0.3s ease-in-out;
    }
    :global(.ink) {
        stroke: black;
        color: black;
        transition:
            fill 1s ease-in-out,
            stroke 1s ease-in-out;
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
        :global(.ink.trees) {
            fill: var(--trees-color);
            stroke: var(--trees-color);
            /* animation: waterVibrate 4s ease-in-out infinite; */
            /* transform-origin: center; */
        }
        :global(.ink.water) {
            fill: var(--water-color);
            stroke: var(--water-color);
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

    @keyframes waterVibrate {
        0% {
            transform: translate(0, 0) rotate(0deg);
        }
        25% {
            transform: translate(-0.5px, 0.5px) rotate(0.5deg);
        }
        50% {
            transform: translate(0.5px, -0.5px) rotate(-0.5deg);
        }
        75% {
            transform: translate(-0.5px, -0.5px) rotate(0deg);
        }
        100% {
            transform: translate(0, 0) rotate(0deg);
        }
    }
</style>
