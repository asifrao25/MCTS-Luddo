/**
 * AI Routes for Luddo AI Engine
 * Handles AI move requests
 */

import { Router, Request, Response } from 'express';
import {
  AIMoveRequest,
  AIMoveResponse,
  MoveDecision,
} from '../types/types.js';
import { createMCTSPlanner } from '../mcts/MCTSPlanner.js';
import { evaluateMoves } from '../evaluation/HeuristicEvaluator.js';
import { DEFAULT_WEIGHTS } from '../types/types.js';

const router = Router();

// Create MCTS planner with 3 second time budget
const mctsPlanner = createMCTSPlanner({
  iterations: 50000,
  timeBudget: 3000,  // 3 seconds max
  maxDepth: 50,
});

/**
 * POST /api/ai/move
 * Get AI move decision for current game state
 */
router.post('/move', async (req: Request, res: Response) => {
  const startTime = performance.now();

  try {
    const { gameState, diceValue, validMoves } = req.body as AIMoveRequest;

    // Validate request
    if (!gameState || !diceValue || !validMoves) {
      const response: AIMoveResponse = {
        success: false,
        error: 'Missing required fields: gameState, diceValue, validMoves',
      };
      return res.status(400).json(response);
    }

    if (!Array.isArray(validMoves) || validMoves.length === 0) {
      const response: AIMoveResponse = {
        success: false,
        error: 'No valid moves provided',
      };
      return res.status(400).json(response);
    }

    // Fast path: only one valid move
    if (validMoves.length === 1) {
      const decision: MoveDecision = {
        tokenId: validMoves[0],
        reasoning: 'Only valid move',
        confidence: 1.0,
        model: 'deterministic',
        evaluationTime: performance.now() - startTime,
        iterations: 0,
      };

      const response: AIMoveResponse = {
        success: true,
        decision,
      };
      return res.json(response);
    }

    // Run MCTS search
    const mctsResult = mctsPlanner.search(gameState, diceValue, validMoves);

    // Enhance reasoning with heuristic insights
    const heuristicDecision = evaluateMoves(
      gameState,
      gameState.currentTurn,
      diceValue,
      validMoves,
      DEFAULT_WEIGHTS
    );

    // Combine MCTS and heuristic reasoning
    let combinedReasoning = mctsResult.reasoning;
    if (heuristicDecision.tokenId === mctsResult.bestMove) {
      combinedReasoning += ` | Heuristic: ${heuristicDecision.reasoning}`;
    } else {
      // MCTS and heuristic disagree - interesting!
      combinedReasoning += ` | MCTS overrides heuristic (${heuristicDecision.reasoning})`;
    }

    const decision: MoveDecision = {
      tokenId: mctsResult.bestMove,
      reasoning: combinedReasoning,
      confidence: mctsResult.confidence,
      model: `mcts-v1 (${mctsResult.iterations} iters)`,
      evaluationTime: mctsResult.evaluationTime,
      iterations: mctsResult.iterations,
    };

    const response: AIMoveResponse = {
      success: true,
      decision,
    };

    console.log(`[AI] Move selected: token ${decision.tokenId} | ${decision.reasoning}`);

    return res.json(response);

  } catch (error) {
    console.error('[AI] Error processing move request:', error);
    const response: AIMoveResponse = {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    };
    return res.status(500).json(response);
  }
});

/**
 * GET /api/ai/health
 * Health check endpoint
 */
router.get('/health', (_req: Request, res: Response) => {
  res.json({
    status: 'healthy',
    model: 'mcts-v1',
    config: mctsPlanner.getConfig(),
  });
});

/**
 * GET /api/ai/status
 * Get AI service status
 */
router.get('/status', (_req: Request, res: Response) => {
  const config = mctsPlanner.getConfig();
  res.json({
    enabled: true,
    model: 'mcts-v1',
    iterations: config.iterations,
    timeBudget: config.timeBudget,
    parallelThreads: config.parallelThreads,
  });
});

export default router;
