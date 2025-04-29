import type { CellType, Unit, RawState, State } from "./types";
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
