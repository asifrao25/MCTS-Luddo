/**
 * Game Logic for Luddo AI Engine
 * Pure functions for game state manipulation
 */

import {
  Player,
  PlayerColor,
  Token,
  OnlineGameState,
  MoveResult,
  PLAYERS_CONFIG,
  START_SPOTS,
  VISIBLE_SAFE_SPOTS
} from '../types/types.js';

/**
 * Calculate global board position from player's step count
 */
export function getGlobalPos(player: PlayerColor, step: number): number {
  if (step === -1) return -1;
  if (step >= 51 && step < 56) return -2; // Home Straight
  if (step >= 56) return 99; // Home

  const startPos = PLAYERS_CONFIG[player].startPos;
  return (startPos + step) % 52;
}

/**
 * Initialize a player with 4 tokens in the yard
 */
export function createPlayer(color: PlayerColor, name: string): Player {
  const config = PLAYERS_CONFIG[color];
  const tokens: Token[] = Array.from({ length: 4 }, (_, i) => ({
    id: i,
    color,
    position: -1,
    stepCount: -1
  }));

  return {
    ...config,
    name,
    tokens,
    active: true
  };
}

/**
 * Calculate valid moves for current player
 */
export function calculateValidMoves(player: Player, diceValue: number): number[] {
  if (!player || !Array.isArray(player.tokens)) return [];
  if (diceValue < 1 || diceValue > 6) return [];

  const validMoves: number[] = [];

  player.tokens.forEach(token => {
    if (token.position === -1) {
      // In yard - can only come out on 6
      if (diceValue === 6) {
        validMoves.push(token.id);
      }
    } else if (token.position !== 99) {
      // On board - can move if won't overshoot
      const boxesLeft = 56 - token.stepCount;
      if (diceValue <= boxesLeft) {
        validMoves.push(token.id);
      }
    }
  });

  return validMoves;
}

/**
 * Execute a token move and return new game state
 */
export function executeTokenMove(
  gameState: OnlineGameState,
  tokenId: number,
  diceValue: number
): MoveResult {
  if (!gameState || !gameState.players || !gameState.currentTurn) {
    return { success: false, error: 'Invalid game state' };
  }

  if (diceValue < 1 || diceValue > 6) {
    return { success: false, error: 'Invalid dice value' };
  }

  if (tokenId < 0 || tokenId > 3) {
    return { success: false, error: 'Invalid token ID' };
  }

  const currentPlayer = gameState.players[gameState.currentTurn];
  if (!currentPlayer) {
    return { success: false, error: 'Invalid current player' };
  }

  const token = currentPlayer.tokens.find(t => t.id === tokenId);
  if (!token) {
    return { success: false, error: 'Token not found' };
  }

  if (!gameState.validMoves.includes(tokenId)) {
    return { success: false, error: 'Invalid move' };
  }

  // Clone state
  const newGameState = structuredClone(gameState) as OnlineGameState;
  const newPlayers = newGameState.players;
  const newToken = newPlayers[gameState.currentTurn].tokens.find(t => t.id === tokenId)!;

  let captured = false;
  let reachedHome = false;
  let capturedToken: { color: PlayerColor; tokenId: number } | undefined;

  if (newToken.position === -1) {
    // Coming out of yard
    newToken.stepCount = 0;
    newToken.position = PLAYERS_CONFIG[gameState.currentTurn].startPos;
  } else {
    // Normal movement
    const newStepCount = newToken.stepCount + diceValue;

    if (newStepCount > 56) {
      return { success: false, error: 'Move would overshoot home' };
    }

    newToken.stepCount = newStepCount;

    if (newToken.stepCount === 56) {
      newToken.position = 99;
      reachedHome = true;
    } else if (newToken.stepCount > 50) {
      newToken.position = 100 + (newToken.stepCount - 51);
    } else {
      newToken.position = getGlobalPos(gameState.currentTurn, newToken.stepCount);
    }
  }

  // Check for captures
  if (
    newToken.position >= 0 &&
    newToken.position < 99 &&
    !START_SPOTS.includes(newToken.position) &&
    !VISIBLE_SAFE_SPOTS.includes(newToken.position)
  ) {
    gameState.activeTurnOrder.forEach(playerColor => {
      if (playerColor === gameState.currentTurn) return;

      newPlayers[playerColor].tokens.forEach((oppToken, idx) => {
        if (oppToken.position === newToken.position) {
          captured = true;
          capturedToken = { color: playerColor, tokenId: oppToken.id };
          newPlayers[playerColor].tokens[idx] = {
            ...oppToken,
            position: -1,
            stepCount: -1
          };
        }
      });
    });
  }

  // Check win condition
  const finishedCount = newPlayers[gameState.currentTurn].tokens.filter(t => t.position === 99).length;
  const gameOver = finishedCount === 4 && !newGameState.rankings.includes(gameState.currentTurn);

  if (gameOver) {
    newGameState.rankings = [...newGameState.rankings, gameState.currentTurn];

    const unfinishedPlayers = gameState.activeTurnOrder.filter(
      p => !newGameState.rankings.includes(p)
    );

    if (unfinishedPlayers.length <= 1) {
      if (unfinishedPlayers.length === 1) {
        newGameState.rankings = [...newGameState.rankings, unfinishedPlayers[0]];
      }
      newGameState.gameState = 'finished';
      newGameState.winner = newGameState.rankings[0];
    }
  }

  // Determine if extra turn
  const shouldGetExtraTurn = diceValue === 6 || captured || reachedHome;

  newGameState.hasRolled = false;
  newGameState.validMoves = [];

  if (!shouldGetExtraTurn && !gameOver) {
    newGameState.currentTurn = getNextPlayer(gameState.currentTurn, gameState.activeTurnOrder, newGameState.rankings);
  }

  return {
    success: true,
    newGameState,
    capturedToken,
    reachedHome,
    gameOver
  };
}

/**
 * Get the next player's turn
 */
export function getNextPlayer(
  currentPlayer: PlayerColor,
  turnOrder: PlayerColor[],
  rankings: PlayerColor[]
): PlayerColor {
  if (!currentPlayer || !turnOrder?.length) {
    return turnOrder?.[0] || 'red';
  }

  const currentIdx = turnOrder.indexOf(currentPlayer);
  if (currentIdx === -1) return turnOrder[0];

  let nextIdx = (currentIdx + 1) % turnOrder.length;
  let attempts = 0;

  while (rankings.includes(turnOrder[nextIdx]) && attempts < turnOrder.length) {
    nextIdx = (nextIdx + 1) % turnOrder.length;
    attempts++;
  }

  if (attempts >= turnOrder.length) {
    return currentPlayer;
  }

  return turnOrder[nextIdx];
}

/**
 * Initialize game state for a new game
 */
export function initializeGameState(
  activePlayers: PlayerColor[],
  playerNames: Record<PlayerColor, string>
): OnlineGameState {
  const players: Record<PlayerColor, Player> = {} as Record<PlayerColor, Player>;

  (['red', 'blue', 'yellow', 'green'] as PlayerColor[]).forEach(color => {
    const isActive = activePlayers.includes(color);
    players[color] = {
      ...createPlayer(color, isActive ? playerNames[color] : PLAYERS_CONFIG[color].name),
      active: isActive
    };
  });

  return {
    gameState: 'playing',
    players,
    activeTurnOrder: activePlayers,
    currentTurn: activePlayers[0],
    diceValue: 1,
    hasRolled: false,
    validMoves: [],
    winner: null,
    rankings: [],
    lastMessage: null
  };
}

/**
 * Roll dice and calculate valid moves
 */
export function rollDice(gameState: OnlineGameState, diceValue?: number): { value: number; validMoves: number[]; autoSkip: boolean } {
  if (!gameState?.players || !gameState.currentTurn) {
    return { value: 1, validMoves: [], autoSkip: true };
  }

  let value = diceValue ?? Math.floor(Math.random() * 6) + 1;

  if (value < 1 || value > 6) {
    value = Math.floor(Math.random() * 6) + 1;
  }

  const currentPlayer = gameState.players[gameState.currentTurn];
  if (!currentPlayer) {
    return { value, validMoves: [], autoSkip: true };
  }

  const validMoves = calculateValidMoves(currentPlayer, value);

  return {
    value,
    validMoves,
    autoSkip: validMoves.length === 0
  };
}
