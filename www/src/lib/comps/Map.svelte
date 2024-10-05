<script lang="ts">
    import { onMount } from "svelte";
    let mapContainer: HTMLElement;

    onMount(async () => {
        if (typeof window !== "undefined") {
            const L = await import("leaflet");
            const map = L.map(mapContainer, {
                center: [51.505, -0.09],
                zoom: 13,
                zoomControl: false,
                dragging: false,
                scrollWheelZoom: false,
                doubleClickZoom: false,
                boxZoom: false,
                keyboard: false,
            });

            L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
                maxZoom: 19,
                attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
            }).addTo(map);
        }
    });
</script>

<div bind:this={mapContainer} style="height: 100%; width: 100%;" />

<style>
    :global(.leaflet-container) {
        height: 100%; /* Full height */
        width: 100%; /* Full width */
        background: none;
    }
</style>
