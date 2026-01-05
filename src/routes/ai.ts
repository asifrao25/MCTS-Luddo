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
import { createParallelMCTS } from '../mcts/ParallelMCTS.js';
import { evaluateMoves } from '../evaluation/HeuristicEvaluator.js';
import { DEFAULT_WEIGHTS } from '../types/types.js';
import { evaluatePosition, isNeuralAvailable } from '../services/NeuralClient.js';

const router = Router();

// Track neural availability
let neuralAvailable = false;
isNeuralAvailable().then(available => {
  neuralAvailable = available;
  console.log(`[AI] Neural evaluator: ${available ? 'AVAILABLE' : 'NOT AVAILABLE'}`);
});

// Create Parallel MCTS planner with 8 threads for M4 Mac
const mctsPlanner = createParallelMCTS({
  iterations: 50000,
  timeBudget: 5500,  // 5.5 seconds for parallel search
  maxDepth: 50,
  parallelThreads: 8,  // M4 has 10 cores, use 8 for MCTS
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

    // Ensure game state has all required fields for MCTS workers
    if (!gameState.rankings) gameState.rankings = [];
    if (gameState.winner === undefined) gameState.winner = null;

    // Ensure all tokens have color property
    for (const playerColor of Object.keys(gameState.players)) {
      const player = gameState.players[playerColor as keyof typeof gameState.players];
      if (player && player.tokens) {
        for (const token of player.tokens) {
          if (!token.color) token.color = playerColor as typeof token.color;
        }
      }
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

    // Run parallel MCTS search and neural evaluation concurrently
    const [mctsResult, neuralScore] = await Promise.all([
      mctsPlanner.search(gameState, diceValue, validMoves),
      neuralAvailable ? evaluatePosition(gameState, gameState.currentTurn) : Promise.resolve(null),
    ]);

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

    // Blend neural evaluation into confidence if available
    let finalConfidence = mctsResult.confidence;
    if (neuralScore !== null) {
      // Blend: 70% MCTS + 30% Neural
      finalConfidence = mctsResult.confidence * 0.7 + neuralScore * 0.3;
      combinedReasoning += ` | Neural: ${(neuralScore * 100).toFixed(1)}%`;
    }

    const decision: MoveDecision = {
      tokenId: mctsResult.bestMove,
      reasoning: combinedReasoning,
      confidence: finalConfidence,
      model: `mcts-v2-parallel (${mctsResult.iterations} iters${neuralScore !== null ? ' +neural' : ''})`,
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
router.get('/health', async (_req: Request, res: Response) => {
  // Check neural availability on each health check
  const neuralStatus = await isNeuralAvailable();
  neuralAvailable = neuralStatus;

  res.json({
    status: 'healthy',
    model: 'mcts-v2-parallel',
    neuralIntegrated: neuralStatus,
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
