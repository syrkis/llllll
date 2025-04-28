import type { CellType, Cell, Unit, RawState, State } from "./types";
export const API_BASE_URL = "http://localhost:8000";

/**
 * Creates a complete API URL by combining the base URL with an endpoint
 * @param endpoint The API endpoint path
 */
export function getApiUrl(endpoint: string): string {
    return `${API_BASE_URL}${endpoint.startsWith("/") ? endpoint : `/${endpoint}`}`;
}
/**
 * Makes an API request and returns the response data
 * @param endpoint The API endpoint path (without base URL)
 * @param method The HTTP method to use
 * @param body Optional request body (will be JSON stringified)
 * @returns The parsed JSON response
 */
export async function apiRequest<T>(
    endpoint: string,
    method = "GET",
    body?: Record<string, unknown>,
): Promise<T> {
    const url = getApiUrl(endpoint);
    const options: RequestInit = { method };

    if (body) {
        options.headers = {
            "Content-Type": "application/json",
        };
        options.body = JSON.stringify(body);
    }

    const response = await fetch(url, options);

    if (!response.ok) {
        throw new Error(
            `API error (${response.status}): ${response.statusText}`,
        );
    }

    const result = await response.json();

    // Check for application-level errors
    if (
        result &&
        typeof result === "object" &&
        "error" in result &&
        result.error
    ) {
        throw new Error(`API error: ${result.error}`);
    }

    return result;
}

/**
 * Maps terrain data from numerical values to cell objects
 */
export function convertTerrain(rawTerrain: number[][]): Cell[][] {
    const terrainMap: Record<number, CellType> = {
        0: "water",
        1: "grass",
        2: "mountain",
    };

    return rawTerrain.map((row, y) =>
        row.map((value, x) => ({
            x,
            y,
            type: terrainMap[value] || "grass",
        })),
    );
}

/**
 * Converts raw unit position data to Unit objects
 */
export function convertUnits(rawPositions: number[]): Unit[][] {
    // This would need to be implemented based on the actual data format
    return rawPositions.map((row, y) =>
        Array.isArray(row)
            ? row.map((unit, x) => ({
                  id: unit.id || Math.floor(Math.random() * 1000),
                  x,
                  y,
                  size: unit.size || 1,
                  health: unit.health || 100,
                  type: unit.type || "unit",
              }))
            : [],
    );
}

/**
 * Transforms raw state data into the application State format
 */
export function parseState(rawState: RawState): State {
    const units = convertUnits(rawState.unit_positions);
    return {
        units,
        pos: rawState.unit_positions,
        step: rawState.step,
    };
}
