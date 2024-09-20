<slot />

<style>
    /* invert if system as darkmode */
    @media (prefers-color-scheme: dark) {
        :global(body) {
            /* filter: invert(1); */
            background-color: black;
            color: white;
        }
    }
    :global(body) {
        margin: 0;
        font-family: Arial, sans-serif;
        font-size: 1.5rem;
        line-height: 1.6;
        text-align: center;
        overflow: hidden;
    }
    :global(h1) {
        margin: 0;
    }
</style>
