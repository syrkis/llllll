import type { State, Unit, UnitType } from "./types";
import { API_BASE_URL, convertTerrain, parseState } from "./utils";

/**
 * Creates a new game based on a location
 * @param place - The location name (e.g., "Copenhagen, Denmark")
 * @returns Promise with the game_id
 */
export async function init(place: string): Promise<string> {
    try {
        // URL encode the place parameter
        const encodedPlace = encodeURIComponent(place);

        // Make the API call to the init endpoint
        const response = await fetch(`${API_BASE_URL}/init/${encodedPlace}`);

        if (!response.ok) {
            throw new Error(
                `Failed to initialize game: ${response.statusText}`,
            );
        }

        const data = await response.json();
        console.log(data);
        return data.game_id;
    } catch (error) {
        console.error("Error initializing game:", error);
        throw error;
    }
}

/**
 * Resets a game with the given game_id
 * @param game_id - The ID of the game to reset
 * @returns Promise with the game state
 */
export async function reset(game_id: string): Promise<State> {
    try {
        const response = await fetch(`${API_BASE_URL}/reset/${game_id}`);

        if (!response.ok) {
            throw new Error(`Failed to reset game: ${response.statusText}`);
        }

        const data = await response.json();

        // Process the state data according to the Python types.py structure
        // The backend uses State with unit_position, unit_health, unit_cooldown
        // We need to transform this to match our frontend State type
        const rawState = data.state as BackendState;

        if (!rawState) {
            throw new Error("No state data returned from the server");
        }

        // Parse the raw state coming from Python/FastAPI to match our TypeScript types
        return {
            units: processUnitData(rawState),
            step: rawState.step || 0,
            pos: Array.isArray(rawState.unit_position)
                ? rawState.unit_position
                : [],
        };
    } catch (error) {
        console.error("Error resetting game:", error);
        throw error;
    }
}

/**
 * Steps the game forward for the given game_id
 * @param game_id - The ID of the game to step
 * @returns Promise with the updated game state
 */
export async function step(game_id: string): Promise<State> {
    try {
        const response = await fetch(`${API_BASE_URL}/step/${game_id}`);

        if (!response.ok) {
            throw new Error(`Failed to step game: ${response.statusText}`);
        }

        const data = await response.json();

        // Process the state data according to the Python types.py structure
        const rawState = data.state as BackendState;

        if (!rawState) {
            throw new Error("No state data returned from the server");
        }

        // Parse the raw state coming from Python/FastAPI to match our TypeScript types
        console.log(rawState);
        return {
            units: processUnitData(rawState),
            step: rawState.step || 0,
            pos: Array.isArray(rawState.unit_position)
                ? rawState.unit_position
                : [],
        };
    } catch (error) {
        console.error("Error stepping game:", error);
        throw error;
    }
}

/**
 * Interface representing raw state data from the API
 * This matches the Python State dataclass in types.py
 */
interface BackendState {
    unit_position?: number[];
    unit_health?: number[];
    unit_cooldown?: number[];
    step?: number;
    [key: string]: unknown;
}

/**
 * Process unit data from the API response
 * @param rawState - The raw state from the API
 * @returns Processed units array matching the frontend types
 */
function processUnitData(rawState: BackendState): Unit[][] {
    // If no unit data is available, return an empty array
    if (!rawState.unit_position || !rawState.unit_health) {
        return [];
    }

    // The backend sends unit_position, unit_health, and unit_cooldown as arrays
    // We need to transform them into our Unit type structure
    const unitPositions = rawState.unit_position;
    const unitHealth = rawState.unit_health || [];
    const unitCooldown = rawState.unit_cooldown || [];

    // Group units by team or other criteria (this will depend on how your data is structured)
    // For now, we'll assume all units are in one group
    const units: Unit[] = [];

    // Process unit data - exact structure depends on how unit_position is formatted
    // This is a basic implementation that assumes unit_position is a flat array of [x1, y1, x2, y2, ...]
    if (Array.isArray(unitPositions) && unitPositions.length >= 2) {
        for (let i = 0; i < unitPositions.length; i += 2) {
            const unitIndex = i / 2;
            units.push({
                id: unitIndex,
                x: unitPositions[i],
                y: unitPositions[i + 1],
                size: 1, // Default size if not provided
                health: unitHealth[unitIndex] || 100, // Default health if not provided
                type: "unit" as UnitType, // Default type
            });
        }
    }

    // Return units grouped as required by the frontend State type
    return [units]; // Wrap in array to match the units: Unit[][] type
}

/**
 * Closes a game with the given game_id
 * @param game_id - The ID of the game to close
 * @returns Promise that resolves when the game is closed
 */
export async function close(game_id: string): Promise<void> {
    try {
        const response = await fetch(`${API_BASE_URL}/close/${game_id}`, {
            method: "POST",
        });

        if (!response.ok) {
            throw new Error(`Failed to close game: ${response.statusText}`);
        }

        // No data processing needed for close operation
    } catch (error) {
        console.error("Error closing game:", error);
        throw error;
    }
}
