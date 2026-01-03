/**
 * Game Simulator for MCTS
 * Provides fast game state simulation for rollouts
 */

import {
  OnlineGameState,
  PlayerColor,
  Player,
  Token,
  PLAYERS_CONFIG,
  START_SPOTS,
  VISIBLE_SAFE_SPOTS
} from '../types/types.js';
import { evaluateMoves } from '../evaluation/HeuristicEvaluator.js';
import { DEFAULT_WEIGHTS } from '../types/types.js';

/**
 * Roll a random dice value
 */
export function rollDice(): number {
  return Math.floor(Math.random() * 6) + 1;
}

/**
 * Calculate global position from step count
 */
function getGlobalPos(player: PlayerColor, step: number): number {
  if (step === -1) return -1;
  if (step >= 51 && step < 56) return -2;
  if (step >= 56) return 99;
  return (PLAYERS_CONFIG[player].startPos + step) % 52;
}

/**
 * Calculate valid moves for a player
 */
export function getValidMoves(player: Player, diceValue: number): number[] {
  const validMoves: number[] = [];

  player.tokens.forEach(token => {
    if (token.position === -1) {
      if (diceValue === 6) validMoves.push(token.id);
    } else if (token.position !== 99) {
      const boxesLeft = 56 - token.stepCount;
      if (diceValue <= boxesLeft) validMoves.push(token.id);
    }
  });

  return validMoves;
}

/**
 * Get next player in turn order
 */
function getNextPlayer(
  current: PlayerColor,
  turnOrder: PlayerColor[],
  rankings: PlayerColor[]
): PlayerColor {
  const idx = turnOrder.indexOf(current);
  let next = (idx + 1) % turnOrder.length;
  let attempts = 0;

  while (rankings.includes(turnOrder[next]) && attempts < turnOrder.length) {
    next = (next + 1) % turnOrder.length;
    attempts++;
  }

  return turnOrder[next];
}

/**
 * Clone game state efficiently
 */
export function cloneState(state: OnlineGameState): OnlineGameState {
  return structuredClone(state);
}

/**
 * Execute a move and return new state (mutates in place for speed)
 */
export function executeMove(
  state: OnlineGameState,
  tokenId: number,
  diceValue: number
): { captured: boolean; reachedHome: boolean; gameOver: boolean } {
  const currentPlayer = state.players[state.currentTurn];
  const token = currentPlayer.tokens.find(t => t.id === tokenId);

  if (!token) return { captured: false, reachedHome: false, gameOver: false };

  let captured = false;
  let reachedHome = false;

  if (token.position === -1) {
    // Exit yard
    token.stepCount = 0;
    token.position = PLAYERS_CONFIG[state.currentTurn].startPos;
  } else {
    // Normal move
    token.stepCount += diceValue;

    if (token.stepCount === 56) {
      token.position = 99;
      reachedHome = true;
    } else if (token.stepCount > 50) {
      token.position = 100 + (token.stepCount - 51);
    } else {
      token.position = getGlobalPos(state.currentTurn, token.stepCount);
    }
  }

  // Check captures
  if (
    token.position >= 0 &&
    token.position < 99 &&
    !START_SPOTS.includes(token.position) &&
    !VISIBLE_SAFE_SPOTS.includes(token.position)
  ) {
    for (const oppColor of state.activeTurnOrder) {
      if (oppColor === state.currentTurn) continue;

      for (const oppToken of state.players[oppColor].tokens) {
        if (oppToken.position === token.position) {
          captured = true;
          oppToken.position = -1;
          oppToken.stepCount = -1;
        }
      }
    }
  }

  // Check win
  const finishedCount = currentPlayer.tokens.filter(t => t.position === 99).length;
  const gameOver = finishedCount === 4 && !state.rankings.includes(state.currentTurn);

  if (gameOver) {
    state.rankings = [...state.rankings, state.currentTurn];

    const unfinished = state.activeTurnOrder.filter(p => !state.rankings.includes(p));
    if (unfinished.length <= 1) {
      if (unfinished.length === 1) {
        state.rankings = [...state.rankings, unfinished[0]];
      }
      state.gameState = 'finished';
      state.winner = state.rankings[0];
    }
  }

  // Determine turn
  const extraTurn = diceValue === 6 || captured || reachedHome;

  if (!extraTurn && !gameOver) {
    state.currentTurn = getNextPlayer(state.currentTurn, state.activeTurnOrder, state.rankings);
  }

  return { captured, reachedHome, gameOver };
}

/**
 * Simulate a random game from current state to completion
 * Uses heuristic-guided rollout for better results
 */
export function simulateRandomGame(
  state: OnlineGameState,
  maxMoves: number = 200
): PlayerColor | null {
  const simState = cloneState(state);
  let moves = 0;

  while (simState.gameState !== 'finished' && moves < maxMoves) {
    const dice = rollDice();
    const player = simState.players[simState.currentTurn];
    const validMoves = getValidMoves(player, dice);

    if (validMoves.length === 0) {
      // No valid moves, switch turn
      simState.currentTurn = getNextPlayer(
        simState.currentTurn,
        simState.activeTurnOrder,
        simState.rankings
      );
      moves++;
      continue;
    }

    // Use heuristic for move selection (better than random)
    let selectedMove: number;

    if (validMoves.length === 1) {
      selectedMove = validMoves[0];
    } else {
      // 70% chance to use heuristic, 30% random for exploration
      if (Math.random() < 0.7) {
        simState.diceValue = dice;
        simState.validMoves = validMoves;
        const decision = evaluateMoves(simState, simState.currentTurn, dice, validMoves, DEFAULT_WEIGHTS);
        selectedMove = decision.tokenId;
      } else {
        selectedMove = validMoves[Math.floor(Math.random() * validMoves.length)];
      }
    }

    executeMove(simState, selectedMove, dice);
    moves++;
  }

  return simState.winner;
}

/**
 * Simulate using pure heuristics (faster for shallow rollouts)
 */
export function simulateHeuristicGame(
  state: OnlineGameState,
  maxMoves: number = 150
): PlayerColor | null {
  const simState = cloneState(state);
  let moves = 0;

  while (simState.gameState !== 'finished' && moves < maxMoves) {
    const dice = rollDice();
    const player = simState.players[simState.currentTurn];
    const validMoves = getValidMoves(player, dice);

    if (validMoves.length === 0) {
      simState.currentTurn = getNextPlayer(
        simState.currentTurn,
        simState.activeTurnOrder,
        simState.rankings
      );
      moves++;
      continue;
    }

    let selectedMove: number;

    if (validMoves.length === 1) {
      selectedMove = validMoves[0];
    } else {
      simState.diceValue = dice;
      simState.validMoves = validMoves;
      const decision = evaluateMoves(simState, simState.currentTurn, dice, validMoves, DEFAULT_WEIGHTS);
      selectedMove = decision.tokenId;
    }

    executeMove(simState, selectedMove, dice);
    moves++;
  }

  return simState.winner;
}

/**
 * Quick position evaluation for leaf nodes
 */
export function quickEvaluate(state: OnlineGameState, forPlayer: PlayerColor): number {
  const player = state.players[forPlayer];
  let score = 0;

  for (const token of player.tokens) {
    if (token.position === 99) {
      score += 100; // Home
    } else if (token.stepCount > 50) {
      score += 60 + (token.stepCount - 50) * 5; // Home stretch
    } else if (token.stepCount >= 0) {
      score += token.stepCount; // Progress
    } else {
      score -= 10; // In yard
    }
  }

  return score;
}
