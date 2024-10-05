import { writable } from "svelte/store";

export const colors = writable({
  teamColors: {
    0: "#888", // gray
    1: "#4b5320", // green
    // You can define more teams if needed
  },
  streakColors: {
    0: "#888", // attack streak colors matching team's color
    1: "#4b5320",
    // Define more if needed
  },
  common: {
    black: "#000",
    white: "#fff",
    // Additional common colors
  },
});
