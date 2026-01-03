/**
 * MCTS Worker - Runs MCTS search in a worker thread
 * Used for parallel tree search
 */

import { parentPort, workerData } from 'worker_threads';
import type { OnlineGameState, PlayerColor } from '../types/types.js';
import {
  cloneState,
  rollDice,
  getValidMoves,
  executeMove,
  simulateHeuristicGame,
  quickEvaluate,
} from './GameSimulator.js';

interface WorkerData {
  gameState: OnlineGameState;
  diceValue: number;
  validMoves: number[];
  iterations: number;
  explorationConstant: number;
  maxDepth: number;
  workerId: number;
}

interface WorkerResult {
  workerId: number;
  moveStats: Map<number, { visits: number; wins: number }>;
  totalIterations: number;
}

// Simple tree node for worker (lighter weight)
class WorkerNode {
  state: OnlineGameState;
  move: number | null;
  visits: number = 0;
  wins: number = 0;
  children: WorkerNode[] = [];
  untriedMoves: number[];
  player: PlayerColor;

  constructor(state: OnlineGameState, move: number | null = null) {
    this.state = state;
    this.move = move;
    this.untriedMoves = [];
    this.player = state.currentTurn;
  }

  uctValue(c: number, parentVisits: number): number {
    if (this.visits === 0) return Infinity;
    return (this.wins / this.visits) + c * Math.sqrt(Math.log(parentVisits) / this.visits);
  }

  isTerminal(): boolean {
    return this.state.gameState === 'finished';
  }
}

function runMCTS(data: WorkerData): WorkerResult {
  const { gameState, diceValue, validMoves, iterations, explorationConstant, maxDepth, workerId } = data;

  // Initialize root
  const root = new WorkerNode(cloneState(gameState));
  root.untriedMoves = [...validMoves];

  let completed = 0;

  for (let i = 0; i < iterations; i++) {
    // Selection
    let node = root;
    let depth = 0;

    while (!node.isTerminal() && node.untriedMoves.length === 0 && node.children.length > 0 && depth < maxDepth) {
      // UCT selection
      let bestChild: WorkerNode | null = null;
      let bestUct = -Infinity;

      for (const child of node.children) {
        const uct = child.uctValue(explorationConstant, node.visits);
        if (uct > bestUct) {
          bestUct = uct;
          bestChild = child;
        }
      }

      if (!bestChild) break;
      node = bestChild;
      depth++;
    }

    // Expansion
    if (!node.isTerminal() && node.untriedMoves.length > 0) {
      const moveIdx = Math.floor(Math.random() * node.untriedMoves.length);
      const move = node.untriedMoves.splice(moveIdx, 1)[0];

      const newState = cloneState(node.state);
      newState.diceValue = diceValue;
      newState.validMoves = getValidMoves(newState.players[newState.currentTurn], diceValue);

      executeMove(newState, move, diceValue);

      const child = new WorkerNode(newState, move);
      child.untriedMoves = getValidMoves(newState.players[newState.currentTurn], rollDice());
      node.children.push(child);
      node = child;
    }

    // Simulation
    let winner: PlayerColor | null;
    if (depth > 15) {
      // Quick evaluation for deep nodes
      let bestPlayer: PlayerColor | null = null;
      let bestScore = -Infinity;
      for (const p of node.state.activeTurnOrder) {
        const score = quickEvaluate(node.state, p);
        if (score > bestScore) {
          bestScore = score;
          bestPlayer = p;
        }
      }
      winner = bestPlayer;
    } else {
      winner = simulateHeuristicGame(node.state, 100);
    }

    // Backpropagation
    let current: WorkerNode | null = node;
    while (current) {
      current.visits++;
      if (winner === current.player) {
        current.wins++;
      } else if (winner === null) {
        current.wins += 0.5;
      }

      // Find parent by traversing from root
      let parent: WorkerNode | null = null;
      const findParent = (n: WorkerNode, target: WorkerNode): WorkerNode | null => {
        for (const child of n.children) {
          if (child === target) return n;
          const found = findParent(child, target);
          if (found) return found;
        }
        return null;
      };

      if (current !== root) {
        parent = findParent(root, current);
      }
      current = parent;
    }

    completed++;
  }

  // Collect stats for root children
  const moveStats = new Map<number, { visits: number; wins: number }>();

  for (const child of root.children) {
    if (child.move !== null) {
      moveStats.set(child.move, {
        visits: child.visits,
        wins: child.wins,
      });
    }
  }

  return {
    workerId,
    moveStats,
    totalIterations: completed,
  };
}

// Worker entry point
if (parentPort) {
  const data = workerData as WorkerData;
  const result = runMCTS(data);

  // Convert Map to object for transfer
  const statsObj: Record<number, { visits: number; wins: number }> = {};
  result.moveStats.forEach((value, key) => {
    statsObj[key] = value;
  });

  parentPort.postMessage({
    workerId: result.workerId,
    moveStats: statsObj,
    totalIterations: result.totalIterations,
  });
}

export { runMCTS, WorkerData, WorkerResult };
