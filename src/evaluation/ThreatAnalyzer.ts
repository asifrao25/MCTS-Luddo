/**
 * Threat Analyzer for Luddo Game
 * Analyzes board position to detect threats and capture opportunities
 */

import {
  PlayerColor,
  Token,
  OnlineGameState,
  ThreatInfo,
  ThreatSource,
  CaptureOpportunity,
  SAFE_SPOTS,
  START_SPOTS,
  PLAYER_START_POSITIONS
} from '../types/types.js';

/**
 * Calculate the global board position from player's step count
 */
function getGlobalPosition(playerColor: PlayerColor, stepCount: number): number {
  if (stepCount < 0) return -1;
  if (stepCount > 50) return -2;
  if (stepCount >= 56) return 99;

  const startPos = PLAYER_START_POSITIONS[playerColor];
  return (startPos + stepCount) % 52;
}

/**
 * Check if a position is a safe spot
 */
function isSafeSpot(position: number): boolean {
  return SAFE_SPOTS.includes(position) || START_SPOTS.includes(position);
}

/**
 * Analyze threats to a player's tokens
 */
export function analyzeThreats(
  gameState: OnlineGameState,
  forPlayer: PlayerColor
): ThreatInfo[] {
  const threats: ThreatInfo[] = [];
  const player = gameState.players[forPlayer];

  for (const token of player.tokens) {
    const threatInfo = analyzeTokenThreat(token, forPlayer, gameState);
    threats.push(threatInfo);
  }

  return threats.sort((a, b) => threatLevelScore(b.threatLevel) - threatLevelScore(a.threatLevel));
}

/**
 * Analyze threat to a single token
 */
function analyzeTokenThreat(
  token: Token,
  playerColor: PlayerColor,
  gameState: OnlineGameState
): ThreatInfo {
  const tokenPos = getGlobalPosition(playerColor, token.stepCount);

  // Tokens in yard, home stretch, or home are safe
  if (tokenPos < 0 || tokenPos === 99 || token.stepCount > 50) {
    return {
      tokenId: token.id,
      threatLevel: 'none',
      threateningSources: [],
      canEscapeWithDice: [],
    };
  }

  // Safe spots are safe
  if (isSafeSpot(tokenPos)) {
    return {
      tokenId: token.id,
      threatLevel: 'none',
      threateningSources: [],
      canEscapeWithDice: [],
    };
  }

  const sources: ThreatSource[] = [];

  // Check each opponent
  for (const oppColor of gameState.activeTurnOrder) {
    if (oppColor === playerColor) continue;

    for (const oppToken of gameState.players[oppColor].tokens) {
      if (oppToken.position < 0 || oppToken.stepCount > 50) continue;

      const oppPos = getGlobalPosition(oppColor, oppToken.stepCount);
      if (oppPos < 0) continue;

      const distance = calculateDistance(oppPos, tokenPos);

      if (distance >= 1 && distance <= 6) {
        sources.push({
          color: oppColor,
          tokenId: oppToken.id,
          distance,
        });
      }
    }
  }

  const threatLevel = calculateThreatLevel(sources);
  const canEscapeWithDice = calculateEscapeDice(token, playerColor, gameState);

  return {
    tokenId: token.id,
    threatLevel,
    threateningSources: sources,
    canEscapeWithDice,
  };
}

/**
 * Calculate distance on circular board
 */
function calculateDistance(fromPos: number, toPos: number): number {
  let distance = (toPos - fromPos + 52) % 52;
  if (distance === 0) distance = 52;
  return distance;
}

/**
 * Calculate threat level from sources
 */
function calculateThreatLevel(sources: ThreatSource[]): ThreatInfo['threatLevel'] {
  if (sources.length === 0) return 'none';

  const minDistance = Math.min(...sources.map(s => s.distance));
  const threatCount = sources.length;

  if (minDistance <= 3 && threatCount >= 2) return 'critical';
  if (minDistance <= 2) return 'high';
  if (minDistance <= 4 && threatCount >= 2) return 'high';
  if (minDistance <= 4) return 'medium';
  return 'low';
}

/**
 * Convert threat level to numeric score
 */
function threatLevelScore(level: ThreatInfo['threatLevel']): number {
  const scores = { critical: 4, high: 3, medium: 2, low: 1, none: 0 };
  return scores[level];
}

/**
 * Calculate which dice values would allow escape
 */
function calculateEscapeDice(
  token: Token,
  playerColor: PlayerColor,
  gameState: OnlineGameState
): number[] {
  const escapeDice: number[] = [];

  for (let dice = 1; dice <= 6; dice++) {
    const newStepCount = token.stepCount + dice;

    if (newStepCount > 56) continue;

    if (newStepCount > 50) {
      escapeDice.push(dice);
      continue;
    }

    const newPos = getGlobalPosition(playerColor, newStepCount);
    if (isSafeSpot(newPos)) {
      escapeDice.push(dice);
    }
  }

  return escapeDice;
}

/**
 * Find capture opportunities for a player with given dice value
 */
export function findCaptureOpportunities(
  gameState: OnlineGameState,
  forPlayer: PlayerColor,
  diceValue: number
): CaptureOpportunity[] {
  const opportunities: CaptureOpportunity[] = [];
  const player = gameState.players[forPlayer];

  for (const token of player.tokens) {
    if (token.position === -1) continue;

    const newStepCount = token.stepCount + diceValue;
    if (newStepCount > 56) continue;
    if (newStepCount > 50) continue;

    const newPos = getGlobalPosition(forPlayer, newStepCount);

    if (isSafeSpot(newPos)) continue;

    for (const oppColor of gameState.activeTurnOrder) {
      if (oppColor === forPlayer) continue;

      for (const oppToken of gameState.players[oppColor].tokens) {
        if (oppToken.position === newPos) {
          opportunities.push({
            tokenId: token.id,
            targetColor: oppColor,
            targetTokenId: oppToken.id,
            targetStepCount: oppToken.stepCount,
            priority: calculateCapturePriority(oppToken, gameState),
          });
        }
      }
    }
  }

  return opportunities.sort((a, b) => b.priority - a.priority);
}

/**
 * Calculate priority of capturing a specific token
 */
function calculateCapturePriority(
  targetToken: Token,
  gameState: OnlineGameState
): number {
  let priority = targetToken.stepCount;

  if (targetToken.stepCount > 40) priority += 30;
  else if (targetToken.stepCount > 30) priority += 15;

  const oppPlayer = gameState.players[targetToken.color];
  const tokensOut = oppPlayer.tokens.filter(
    t => t.position >= 0 && t.position < 99 && t.stepCount <= 50
  ).length;

  if (tokensOut <= 1) priority += 25;
  else if (tokensOut === 2) priority += 10;

  return priority;
}

/**
 * Check if a token would be under threat after moving
 */
export function wouldBeUnderThreat(
  token: Token,
  playerColor: PlayerColor,
  diceValue: number,
  gameState: OnlineGameState
): boolean {
  const newStepCount = token.stepCount + diceValue;

  if (newStepCount > 50) return false;

  const newPos = getGlobalPosition(playerColor, newStepCount);

  if (isSafeSpot(newPos)) return false;

  for (const oppColor of gameState.activeTurnOrder) {
    if (oppColor === playerColor) continue;

    for (const oppToken of gameState.players[oppColor].tokens) {
      if (oppToken.position < 0 || oppToken.stepCount > 50) continue;

      const oppPos = getGlobalPosition(oppColor, oppToken.stepCount);
      if (oppPos < 0) continue;

      const distance = calculateDistance(oppPos, newPos);
      if (distance >= 1 && distance <= 6) {
        return true;
      }
    }
  }

  return false;
}
