/**
 * MCTS Planner - Monte Carlo Tree Search for Luddo
 * Main search algorithm with UCT selection
 */

import {
  OnlineGameState,
  PlayerColor,
  MCTSConfig,
  MCTSResult,
  DEFAULT_MCTS_CONFIG,
} from '../types/types.js';
import { TreeNode } from './TreeNode.js';
import {
  cloneState,
  rollDice,
  getValidMoves,
  executeMove,
  simulateHeuristicGame,
  quickEvaluate,
} from './GameSimulator.js';

/**
 * MCTS Planner class
 */
export class MCTSPlanner {
  private config: MCTSConfig;

  constructor(config: Partial<MCTSConfig> = {}) {
    this.config = { ...DEFAULT_MCTS_CONFIG, ...config };
  }

  /**
   * Search for the best move using MCTS
   */
  search(
    gameState: OnlineGameState,
    diceValue: number,
    validMoves: number[]
  ): MCTSResult {
    const startTime = performance.now();
    const aiPlayer = gameState.currentTurn;

    // Create root node
    const root = new TreeNode(cloneState(gameState));
    root.untriedMoves = [...validMoves];

    let iterations = 0;
    const timeBudget = this.config.timeBudget;

    // Main MCTS loop
    while (iterations < this.config.iterations) {
      // Check time budget
      if (performance.now() - startTime > timeBudget) break;

      // Selection: traverse tree using UCT
      let node = this.select(root);

      // Expansion: add new child if not terminal
      if (!node.isTerminal() && !node.isFullyExpanded()) {
        node = this.expand(node, diceValue);
      }

      // Simulation: play out to terminal state
      const winner = this.simulate(node, aiPlayer);

      // Backpropagation: update statistics
      node.backpropagate(winner);

      iterations++;
    }

    const elapsed = performance.now() - startTime;

    // Select best move (most visited)
    const bestChild = root.selectMostVisitedChild();

    if (!bestChild || bestChild.move === null) {
      // Fallback to first valid move
      return {
        bestMove: validMoves[0],
        confidence: 0.5,
        iterations,
        evaluationTime: elapsed,
        reasoning: 'MCTS fallback - no clear best move',
      };
    }

    // Generate reasoning from search statistics
    const reasoning = this.generateReasoning(root, bestChild, iterations);

    return {
      bestMove: bestChild.move,
      confidence: this.calculateConfidence(root, bestChild),
      iterations,
      evaluationTime: elapsed,
      reasoning,
    };
  }

  /**
   * Selection phase - traverse tree using UCT
   */
  private select(node: TreeNode): TreeNode {
    while (!node.isTerminal()) {
      if (!node.isFullyExpanded()) {
        return node;
      }

      const bestChild = node.selectBestChild(this.config.explorationConstant);
      if (!bestChild) return node;
      node = bestChild;

      // Limit depth
      if (node.depth() >= this.config.maxDepth) return node;
    }

    return node;
  }

  /**
   * Expansion phase - add a new child node
   */
  private expand(node: TreeNode, diceValue: number): TreeNode {
    if (node.untriedMoves.length === 0) return node;

    // Select random untried move
    const moveIdx = Math.floor(Math.random() * node.untriedMoves.length);
    const move = node.untriedMoves[moveIdx];

    // Create new state
    const newState = cloneState(node.state);
    newState.diceValue = diceValue;

    // Ensure validMoves is set
    const player = newState.players[newState.currentTurn];
    newState.validMoves = getValidMoves(player, diceValue);

    // Execute move
    executeMove(newState, move, diceValue);

    // Create and add child
    return node.addChild(newState, move, diceValue);
  }

  /**
   * Simulation phase - play out game to completion
   */
  private simulate(node: TreeNode, aiPlayer: PlayerColor): PlayerColor | null {
    // For deep nodes, use quick evaluation instead of full rollout
    if (node.depth() > 20) {
      return this.quickSimulate(node.state, aiPlayer);
    }

    return simulateHeuristicGame(node.state);
  }

  /**
   * Quick simulation using position evaluation
   */
  private quickSimulate(state: OnlineGameState, aiPlayer: PlayerColor): PlayerColor | null {
    // Compare scores between all active players
    let bestPlayer: PlayerColor | null = null;
    let bestScore = -Infinity;

    for (const playerColor of state.activeTurnOrder) {
      const score = quickEvaluate(state, playerColor);
      if (score > bestScore) {
        bestScore = score;
        bestPlayer = playerColor;
      }
    }

    return bestPlayer;
  }

  /**
   * Calculate confidence from search results
   */
  private calculateConfidence(root: TreeNode, bestChild: TreeNode): number {
    if (root.visits === 0) return 0.5;

    // Base confidence on win rate and visit ratio
    const winRate = bestChild.winRate();
    const visitRatio = bestChild.visits / root.visits;

    // High confidence if both win rate and visit ratio are high
    const confidence = 0.5 + (winRate * 0.3) + (visitRatio * 0.2);

    return Math.min(0.99, Math.max(0.3, confidence));
  }

  /**
   * Generate reasoning from search statistics
   */
  private generateReasoning(root: TreeNode, bestChild: TreeNode, iterations: number): string {
    const parts: string[] = [];

    // Win rate info
    const winRate = (bestChild.winRate() * 100).toFixed(1);
    parts.push(`Win rate: ${winRate}%`);

    // Visit info
    parts.push(`Visits: ${bestChild.visits}/${root.visits}`);

    // Iterations
    parts.push(`${iterations} iterations`);

    // Move quality assessment
    if (bestChild.winRate() > 0.7) {
      parts.push('Strong move');
    } else if (bestChild.winRate() > 0.5) {
      parts.push('Good move');
    } else if (bestChild.winRate() > 0.3) {
      parts.push('Acceptable move');
    } else {
      parts.push('Difficult position');
    }

    return parts.join(' | ');
  }

  /**
   * Update configuration
   */
  setConfig(config: Partial<MCTSConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Get current configuration
   */
  getConfig(): MCTSConfig {
    return { ...this.config };
  }
}

/**
 * Create a new MCTS planner with default config
 */
export function createMCTSPlanner(config?: Partial<MCTSConfig>): MCTSPlanner {
  return new MCTSPlanner(config);
}
