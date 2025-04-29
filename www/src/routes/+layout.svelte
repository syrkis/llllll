<slot />

<style>
    :global(:root) {
        --red: #c62828; /* this one is for folke */
        --blue: #283593;
        --green: #43a047;
        --transition: 700ms ease;
    }
    :global(body) {
        margin: 0;
        padding: 0;
        overflow: hidden;
        font-size: 1.5em;
        line-height: 2;
        font-family: "Libertinus Serif", serif;
        transition:
            background var(--transition),
            color var(--transition);
    }
    :global(circle) {
        transition:
            cx var(--transition),
            cy var(--transition),
            r var(--transition),
            fill var(--transition);
    }

    :global(rect) {
        transition:
            x var(--transition),
            y var(--transition),
            width var(--transition),
            height var(--transition),
            fill var(--transition);
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
