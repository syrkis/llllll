<slot />

<style>
    :global(:root) {
        --red: #c62828; /* this one is for folke */
        --blue: #283593;
    }
    :global(body) {
        font-family: "Libertinus Serif", serif;
    }
    :global(circle, rect, svg) {
        transition: all 500ms ease-in-out;
    }

    @media (prefers-color-scheme: dark) {
        :global(body) {
            color: white;
            background: black;
        }
        :global(svg) {
            fill: white;
        }
    }
</style>
