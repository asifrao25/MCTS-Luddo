/**
 * Parallel MCTS - Coordinates multiple worker threads for MCTS search
 * Uses root parallelization strategy
 */

import { Worker } from 'worker_threads';
import { join } from 'path';
import type {
  OnlineGameState,
  PlayerColor,
  MCTSConfig,
  MCTSResult,
} from '../types/types.js';

// Get directory for worker file
const getWorkerPath = (): string => {
  // In compiled output, workers are in the same directory
  return join(process.cwd(), 'dist', 'mcts', 'MCTSWorker.js');
};

interface WorkerMessage {
  workerId: number;
  moveStats: Record<number, { visits: number; wins: number }>;
  totalIterations: number;
}

/**
 * Parallel MCTS Planner using worker threads
 */
export class ParallelMCTS {
  private config: MCTSConfig;
  private numWorkers: number;

  constructor(config: Partial<MCTSConfig> = {}) {
    this.config = {
      iterations: 50000,
      explorationConstant: Math.sqrt(2),
      maxDepth: 50,
      timeBudget: 5500,
      parallelThreads: 8,
      ...config,
    };
    this.numWorkers = Math.min(this.config.parallelThreads, 8);
  }

  /**
   * Run parallel MCTS search
   */
  async search(
    gameState: OnlineGameState,
    diceValue: number,
    validMoves: number[]
  ): Promise<MCTSResult> {
    const startTime = performance.now();

    // Fast path: only one valid move
    if (validMoves.length === 1) {
      return {
        bestMove: validMoves[0],
        confidence: 1.0,
        iterations: 0,
        evaluationTime: performance.now() - startTime,
        reasoning: 'Only valid move',
      };
    }

    // Distribute iterations across workers
    const iterationsPerWorker = Math.floor(this.config.iterations / this.numWorkers);

    // Spawn workers and collect results
    const workerPromises: Promise<WorkerMessage>[] = [];

    for (let i = 0; i < this.numWorkers; i++) {
      workerPromises.push(this.runWorker(i, gameState, diceValue, validMoves, iterationsPerWorker));
    }

    // Wait for all workers with timeout
    const timeoutPromise = new Promise<WorkerMessage[]>((_, reject) => {
      setTimeout(() => reject(new Error('Worker timeout')), this.config.timeBudget + 1000);
    });

    let results: WorkerMessage[];
    try {
      results = await Promise.race([
        Promise.all(workerPromises),
        timeoutPromise,
      ]) as WorkerMessage[];
    } catch (error) {
      console.warn('[ParallelMCTS] Workers timed out, using partial results');
      // Collect any completed results
      results = [];
      for (const p of workerPromises) {
        try {
          const r = await Promise.race([p, new Promise<null>((resolve) => setTimeout(() => resolve(null), 100))]);
          if (r) results.push(r);
        } catch {
          // Worker didn't complete
        }
      }
    }

    const elapsed = performance.now() - startTime;

    // Aggregate results from all workers
    const aggregatedStats = this.aggregateResults(results, validMoves);

    // Find best move (most visits)
    let bestMove = validMoves[0];
    let bestVisits = 0;
    let totalIterations = 0;

    for (const [move, stats] of aggregatedStats.entries()) {
      totalIterations += stats.visits;
      if (stats.visits > bestVisits) {
        bestVisits = stats.visits;
        bestMove = move;
      }
    }

    // Calculate confidence
    const bestStats = aggregatedStats.get(bestMove);
    const winRate = bestStats ? bestStats.wins / bestStats.visits : 0.5;
    const confidence = Math.min(0.99, 0.5 + winRate * 0.4);

    // Generate reasoning
    const reasoning = this.generateReasoning(aggregatedStats, bestMove, totalIterations, this.numWorkers);

    return {
      bestMove,
      confidence,
      iterations: totalIterations,
      evaluationTime: elapsed,
      reasoning,
    };
  }

  /**
   * Run a single worker
   */
  private runWorker(
    workerId: number,
    gameState: OnlineGameState,
    diceValue: number,
    validMoves: number[],
    iterations: number
  ): Promise<WorkerMessage> {
    return new Promise((resolve, reject) => {
      const workerPath = getWorkerPath();

      const worker = new Worker(workerPath, {
        workerData: {
          gameState,
          diceValue,
          validMoves,
          iterations,
          explorationConstant: this.config.explorationConstant,
          maxDepth: this.config.maxDepth,
          workerId,
        },
      });

      const timeout = setTimeout(() => {
        worker.terminate();
        reject(new Error(`Worker ${workerId} timed out`));
      }, this.config.timeBudget);

      worker.on('message', (result: WorkerMessage) => {
        clearTimeout(timeout);
        worker.terminate();
        resolve(result);
      });

      worker.on('error', (error) => {
        clearTimeout(timeout);
        worker.terminate();
        reject(error);
      });

      worker.on('exit', (code) => {
        if (code !== 0) {
          clearTimeout(timeout);
          reject(new Error(`Worker ${workerId} exited with code ${code}`));
        }
      });
    });
  }

  /**
   * Aggregate results from multiple workers
   */
  private aggregateResults(
    results: WorkerMessage[],
    validMoves: number[]
  ): Map<number, { visits: number; wins: number }> {
    const aggregated = new Map<number, { visits: number; wins: number }>();

    // Initialize all valid moves
    for (const move of validMoves) {
      aggregated.set(move, { visits: 0, wins: 0 });
    }

    // Sum up results from all workers
    for (const result of results) {
      for (const [moveStr, stats] of Object.entries(result.moveStats)) {
        const move = parseInt(moveStr, 10);
        const current = aggregated.get(move);
        if (current) {
          current.visits += stats.visits;
          current.wins += stats.wins;
        }
      }
    }

    return aggregated;
  }

  /**
   * Generate reasoning from aggregated stats
   */
  private generateReasoning(
    stats: Map<number, { visits: number; wins: number }>,
    bestMove: number,
    totalIterations: number,
    numWorkers: number
  ): string {
    const parts: string[] = [];

    const bestStats = stats.get(bestMove);
    if (bestStats && bestStats.visits > 0) {
      const winRate = ((bestStats.wins / bestStats.visits) * 100).toFixed(1);
      parts.push(`Win rate: ${winRate}%`);
      parts.push(`Visits: ${bestStats.visits}`);
    }

    parts.push(`${totalIterations} iterations`);
    parts.push(`${numWorkers} threads`);

    // Quality assessment
    const winRate = bestStats ? bestStats.wins / bestStats.visits : 0;
    if (winRate > 0.7) {
      parts.push('Strong move');
    } else if (winRate > 0.5) {
      parts.push('Good move');
    } else if (winRate > 0.3) {
      parts.push('Reasonable move');
    } else {
      parts.push('Challenging position');
    }

    return parts.join(' | ');
  }

  /**
   * Update configuration
   */
  setConfig(config: Partial<MCTSConfig>): void {
    this.config = { ...this.config, ...config };
    this.numWorkers = Math.min(this.config.parallelThreads, 8);
  }

  /**
   * Get current configuration
   */
  getConfig(): MCTSConfig {
    return { ...this.config };
  }
}

/**
 * Create a parallel MCTS planner
 */
export function createParallelMCTS(config?: Partial<MCTSConfig>): ParallelMCTS {
  return new ParallelMCTS(config);
}
