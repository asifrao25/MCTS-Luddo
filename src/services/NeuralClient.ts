/**
 * Neural Evaluator Client
 * Calls the Python MLX neural network for position evaluation
 */

import type { OnlineGameState, PlayerColor } from '../types/types.js';

const NEURAL_URL = 'http://localhost:3021';
const TIMEOUT_MS = 500; // Fast timeout for neural calls

interface NeuralResponse {
  success: boolean;
  score?: number;
  error?: string;
}

/**
 * Evaluate a position using the neural network
 * Returns win probability for the specified player (0-1)
 */
export async function evaluatePosition(
  gameState: OnlineGameState,
  playerColor: PlayerColor
): Promise<number | null> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const response = await fetch(`${NEURAL_URL}/evaluate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ gameState, playerColor }),
      signal: controller.signal,
    });

    if (!response.ok) {
      console.warn('[NeuralClient] Non-OK response:', response.status);
      return null;
    }

    const data = await response.json() as NeuralResponse;

    if (data.success && typeof data.score === 'number') {
      return data.score;
    }

    return null;
  } catch (error: unknown) {
    // Silently fail - neural is optional
    if (error instanceof Error && error.name !== 'AbortError') {
      console.warn('[NeuralClient] Error:', error.message);
    }
    return null;
  } finally {
    clearTimeout(timeout);
  }
}

/**
 * Batch evaluate multiple positions
 * Returns array of scores or null on failure
 */
export async function batchEvaluate(
  states: Array<{ gameState: OnlineGameState; playerColor: PlayerColor }>
): Promise<number[] | null> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), TIMEOUT_MS * 2);

  try {
    const response = await fetch(`${NEURAL_URL}/batch_evaluate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ states }),
      signal: controller.signal,
    });

    if (!response.ok) {
      return null;
    }

    const data = await response.json() as { success: boolean; scores?: number[] };

    if (data.success && Array.isArray(data.scores)) {
      return data.scores;
    }

    return null;
  } catch {
    return null;
  } finally {
    clearTimeout(timeout);
  }
}

/**
 * Check if neural evaluator is available
 */
export async function isNeuralAvailable(): Promise<boolean> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 200);

  try {
    const response = await fetch(`${NEURAL_URL}/health`, {
      signal: controller.signal,
    });

    if (!response.ok) return false;

    const data = await response.json() as { status: string; model_loaded: boolean };
    return data.status === 'healthy' && data.model_loaded === true;
  } catch {
    return false;
  } finally {
    clearTimeout(timeout);
  }
}
