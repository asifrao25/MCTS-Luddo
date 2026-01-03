/**
 * Heuristic Evaluator for Luddo Game
 * Evaluates moves and board positions using strategic heuristics
 */

import {
  PlayerColor,
  Token,
  OnlineGameState,
  MoveCandidate,
  MoveDecision,
  MoveTag,
  EvaluationWeights,
  PositionEvaluation,
  DEFAULT_WEIGHTS,
  SAFE_SPOTS,
  START_SPOTS,
  PLAYER_START_POSITIONS,
} from '../types/types.js';
import {
  analyzeThreats,
  findCaptureOpportunities,
  wouldBeUnderThreat,
} from './ThreatAnalyzer.js';

/**
 * Calculate global position from step count
 */
function getGlobalPosition(playerColor: PlayerColor, stepCount: number): number {
  if (stepCount < 0) return -1;
  if (stepCount > 50) return -2;
  if (stepCount >= 56) return 99;

  const startPos = PLAYER_START_POSITIONS[playerColor];
  return (startPos + stepCount) % 52;
}

/**
 * Check if position is safe
 */
function isSafeSpot(position: number): boolean {
  return SAFE_SPOTS.includes(position) || START_SPOTS.includes(position);
}

/**
 * Evaluate all valid moves and return the best one
 */
export function evaluateMoves(
  gameState: OnlineGameState,
  playerColor: PlayerColor,
  diceValue: number,
  validMoves: number[],
  weights: EvaluationWeights = DEFAULT_WEIGHTS
): MoveDecision {
  const player = gameState.players[playerColor];
  const threats = analyzeThreats(gameState, playerColor);
  const captureOpportunities = findCaptureOpportunities(gameState, playerColor, diceValue);

  const candidates: MoveCandidate[] = [];

  for (const tokenId of validMoves) {
    const token = player.tokens.find(t => t.id === tokenId);
    if (!token) continue;

    const candidate = evaluateSingleMove(
      token,
      playerColor,
      diceValue,
      gameState,
      threats,
      captureOpportunities,
      weights
    );

    candidates.push(candidate);
  }

  candidates.sort((a, b) => b.score - a.score);

  const best = candidates[0];

  return {
    tokenId: best.tokenId,
    reasoning: best.reasoning,
    confidence: calculateConfidence(candidates),
    model: 'heuristic-v1',
  };
}

/**
 * Evaluate a single move
 */
function evaluateSingleMove(
  token: Token,
  playerColor: PlayerColor,
  diceValue: number,
  gameState: OnlineGameState,
  threats: ReturnType<typeof analyzeThreats>,
  captureOpportunities: ReturnType<typeof findCaptureOpportunities>,
  weights: EvaluationWeights
): MoveCandidate {
  let score = 0;
  const tags: MoveTag[] = [];
  const reasons: string[] = [];

  const player = gameState.players[playerColor];
  const tokenThreat = threats.find(t => t.tokenId === token.id);

  // === SPECIAL CASE: Exiting yard ===
  if (token.position === -1 && diceValue === 6) {
    score += weights.exitYardBonus + 40;
    tags.push('exit-yard');
    reasons.push('Exit yard');

    const tokensOut = player.tokens.filter(t => t.position >= 0 && t.position < 99).length;
    if (tokensOut === 0) {
      score += 30;
      reasons.push('First token out!');
    }

    return {
      tokenId: token.id,
      score,
      reasoning: reasons.join(', '),
      tags,
    };
  }

  // Skip if token is in yard and dice isn't 6
  if (token.position === -1) {
    return {
      tokenId: token.id,
      score: -1000,
      reasoning: 'Cannot move from yard',
      tags: [],
    };
  }

  const newStepCount = token.stepCount + diceValue;

  // === HIGHEST PRIORITY: Reaching home ===
  if (newStepCount === 56) {
    score += weights.finishedToken + 100;
    tags.push('reach-home');
    reasons.push('REACH HOME!');

    const finishedCount = player.tokens.filter(t => t.position === 99).length;
    if (finishedCount === 3) {
      score += 500;
      reasons.push('WINNING MOVE!');
    }

    return {
      tokenId: token.id,
      score,
      reasoning: reasons.join(', '),
      tags,
    };
  }

  // === HIGH PRIORITY: Capture opportunity ===
  const capture = captureOpportunities.find(c => c.tokenId === token.id);
  if (capture) {
    score += weights.captureBonus + capture.priority;
    tags.push('capture');
    reasons.push(`Capture ${capture.targetColor} token (+${capture.priority})`);
  }

  // === HIGH PRIORITY: Escape from threat ===
  if (tokenThreat && tokenThreat.threatLevel !== 'none') {
    const newPos = getGlobalPosition(playerColor, newStepCount);
    const escapesToSafety = newStepCount > 50 || isSafeSpot(newPos);

    if (escapesToSafety) {
      const escapeBonus =
        tokenThreat.threatLevel === 'critical' ? 60 :
        tokenThreat.threatLevel === 'high' ? 40 :
        tokenThreat.threatLevel === 'medium' ? 25 : 10;

      score += escapeBonus;
      tags.push('escape');
      reasons.push(`Escape ${tokenThreat.threatLevel} threat`);
    }
  }

  // === MEDIUM PRIORITY: Enter home stretch ===
  if (newStepCount > 50 && token.stepCount <= 50) {
    score += weights.homeStretchBonus;
    tags.push('home-stretch');
    reasons.push('Enter home stretch');
  }

  // === MEDIUM PRIORITY: Advance in home stretch ===
  if (token.stepCount > 50 && newStepCount > token.stepCount) {
    score += 30 + (newStepCount - 50) * 5;
    reasons.push('Advance in home stretch');
  }

  // === MEDIUM PRIORITY: Land on safe spot ===
  if (newStepCount <= 50) {
    const newPos = getGlobalPosition(playerColor, newStepCount);
    if (isSafeSpot(newPos)) {
      score += weights.safeSpotBonus;
      tags.push('safe-spot');
      reasons.push('Land on safe spot');
    }
  }

  // === BASE: Progress score ===
  score += token.stepCount * weights.tokenProgress / 10;
  score += diceValue * 2;

  // === PENALTY: Moving into danger ===
  if (newStepCount <= 50 && !capture) {
    const underThreat = wouldBeUnderThreat(token, playerColor, diceValue, gameState);
    if (underThreat) {
      const newPos = getGlobalPosition(playerColor, newStepCount);
      if (!isSafeSpot(newPos)) {
        score += weights.threatPenalty;
        tags.push('risky');
        reasons.push('Moves into danger');
      }
    }
  }

  if (reasons.length === 0) {
    tags.push('advance');
    reasons.push('Advance token');
  }

  return {
    tokenId: token.id,
    score,
    reasoning: reasons.join(', '),
    tags,
  };
}

/**
 * Calculate confidence based on score difference between top moves
 */
function calculateConfidence(candidates: MoveCandidate[]): number {
  if (candidates.length <= 1) return 1.0;

  const best = candidates[0].score;
  const second = candidates[1].score;
  const diff = best - second;

  if (diff > 50) return 0.95;
  if (diff > 30) return 0.85;
  if (diff > 15) return 0.75;
  if (diff > 5) return 0.65;
  return 0.55;
}

/**
 * Evaluate overall board position for a player
 */
export function evaluatePosition(
  gameState: OnlineGameState,
  playerColor: PlayerColor,
  weights: EvaluationWeights = DEFAULT_WEIGHTS
): PositionEvaluation {
  const player = gameState.players[playerColor];
  const threats = analyzeThreats(gameState, playerColor);

  let progress = 0;
  let safety = 0;
  let threatScore = 0;
  let opportunities = 0;

  for (const token of player.tokens) {
    if (token.position === 99) {
      progress += weights.finishedToken;
    } else if (token.stepCount > 50) {
      progress += weights.homeStretchBonus + (token.stepCount - 50) * 10;
    } else if (token.stepCount >= 0) {
      progress += token.stepCount * weights.tokenProgress;
      progress += weights.exitYardBonus;
    } else {
      progress += weights.yardPenalty;
    }

    if (token.position >= 0 && token.position < 99 && token.stepCount <= 50) {
      const pos = getGlobalPosition(playerColor, token.stepCount);
      if (isSafeSpot(pos)) {
        safety += weights.safeSpotBonus;
      }
    }

    const tokenThreat = threats.find(t => t.tokenId === token.id);
    if (tokenThreat && tokenThreat.threatLevel !== 'none') {
      const penalty =
        tokenThreat.threatLevel === 'critical' ? weights.threatPenalty * 2 :
        tokenThreat.threatLevel === 'high' ? weights.threatPenalty * 1.5 :
        tokenThreat.threatLevel === 'medium' ? weights.threatPenalty :
        weights.threatPenalty * 0.5;

      threatScore += penalty;
    }
  }

  for (let dice = 1; dice <= 6; dice++) {
    const captures = findCaptureOpportunities(gameState, playerColor, dice);
    opportunities += captures.length * 5;
  }

  return {
    totalScore: progress + safety + threatScore + opportunities,
    breakdown: {
      progress,
      safety,
      threats: threatScore,
      opportunities,
    },
  };
}
