/**
 * Types for Luddo AI Engine Backend
 */

// === Core Game Types ===

export type PlayerColor = 'red' | 'green' | 'yellow' | 'blue';
export type GameState = 'setup' | 'playing' | 'finished';

export interface Token {
  id: number;
  color: PlayerColor;
  position: number;        // -1=yard, 0-51=main track, 100+=home stretch, 99=finished
  stepCount: number;       // -1=yard, 0-56=steps from start
}

export interface Player {
  id: PlayerColor;
  name: string;
  color: string;
  darkColor: string;
  bgClass: string;
  textClass: string;
  borderClass: string;
  tokens: Token[];
  startPos: number;
  active: boolean;
  isAI?: boolean;
  kills: number;
  sixes: number;
}

export interface OnlineGameState {
  gameState: GameState;
  players: Record<PlayerColor, Player>;
  activeTurnOrder: PlayerColor[];
  currentTurn: PlayerColor;
  diceValue: number;
  hasRolled: boolean;
  validMoves: number[];
  winner: PlayerColor | null;
  rankings: PlayerColor[];
  lastMessage: string | null;
}

export interface MoveResult {
  success: boolean;
  newGameState?: OnlineGameState;
  capturedToken?: { color: PlayerColor; tokenId: number };
  reachedHome?: boolean;
  gameOver?: boolean;
  error?: string;
}

// === AI Types ===

export interface MoveDecision {
  tokenId: number;
  reasoning: string;
  confidence: number;
  model: string;
  evaluationTime?: number;
  iterations?: number;
}

export interface MoveCandidate {
  tokenId: number;
  score: number;
  reasoning: string;
  tags: MoveTag[];
}

export type MoveTag =
  | 'exit-yard'
  | 'capture'
  | 'escape'
  | 'safe-spot'
  | 'home-stretch'
  | 'reach-home'
  | 'advance'
  | 'risky'
  | 'blocking';

export interface ThreatInfo {
  tokenId: number;
  threatLevel: 'critical' | 'high' | 'medium' | 'low' | 'none';
  threateningSources: ThreatSource[];
  canEscapeWithDice: number[];
}

export interface ThreatSource {
  color: PlayerColor;
  tokenId: number;
  distance: number;
}

export interface CaptureOpportunity {
  tokenId: number;
  targetColor: PlayerColor;
  targetTokenId: number;
  targetStepCount: number;
  priority: number;
}

export interface PositionEvaluation {
  totalScore: number;
  breakdown: {
    progress: number;
    safety: number;
    threats: number;
    opportunities: number;
  };
}

export interface EvaluationWeights {
  tokenProgress: number;
  finishedToken: number;
  homeStretchBonus: number;
  exitYardBonus: number;
  yardPenalty: number;
  safeSpotBonus: number;
  threatPenalty: number;
  captureBonus: number;
  blockingBonus: number;
}

// === MCTS Types ===

export interface MCTSConfig {
  iterations: number;
  explorationConstant: number;
  maxDepth: number;
  timeBudget: number;
  parallelThreads: number;
}

export interface MCTSNode {
  state: OnlineGameState;
  parent: MCTSNode | null;
  children: MCTSNode[];
  move: number | null;       // tokenId that led to this state
  diceValue: number | null;  // dice value for this move
  visits: number;
  wins: number;
  isChanceNode: boolean;     // Dice roll node
  untriedMoves: number[];
}

export interface MCTSResult {
  bestMove: number;
  confidence: number;
  iterations: number;
  evaluationTime: number;
  reasoning: string;
}

// === API Types ===

export interface AIMoveRequest {
  gameState: OnlineGameState;
  diceValue: number;
  validMoves: number[];
}

export interface AIMoveResponse {
  success: boolean;
  decision?: MoveDecision;
  error?: string;
}

// === Constants ===

export const SAFE_SPOTS = [0, 8, 13, 21, 26, 34, 39, 47];
export const START_SPOTS = [0, 13, 26, 39];
export const VISIBLE_SAFE_SPOTS = [8, 21, 34, 47];

export const PLAYER_START_POSITIONS: Record<PlayerColor, number> = {
  red: 0,
  blue: 13,
  yellow: 26,
  green: 39,
};

export const DEFAULT_WEIGHTS: EvaluationWeights = {
  tokenProgress: 10,
  finishedToken: 200,
  homeStretchBonus: 50,
  exitYardBonus: 40,
  yardPenalty: -15,
  safeSpotBonus: 15,
  threatPenalty: -25,
  captureBonus: 30,
  blockingBonus: 20,
};

export const DEFAULT_MCTS_CONFIG: MCTSConfig = {
  iterations: 50000,
  explorationConstant: Math.sqrt(2),
  maxDepth: 50,
  timeBudget: 3000,  // 3 seconds max (was 5.5s)
  parallelThreads: 8,
};

export const PLAYERS_CONFIG: Record<PlayerColor, Omit<Player, 'tokens' | 'active'>> = {
  red: {
    id: 'red',
    name: 'Red',
    color: '#ef4444',
    darkColor: '#991b1b',
    bgClass: 'bg-red-500',
    textClass: 'text-red-500',
    borderClass: 'border-red-500',
    startPos: 0,
    kills: 0,
    sixes: 0
  },
  blue: {
    id: 'blue',
    name: 'Blue',
    color: '#3b82f6',
    darkColor: '#1e40af',
    bgClass: 'bg-blue-500',
    textClass: 'text-blue-500',
    borderClass: 'border-blue-500',
    startPos: 13,
    kills: 0,
    sixes: 0
  },
  yellow: {
    id: 'yellow',
    name: 'Yellow',
    color: '#eab308',
    darkColor: '#854d0e',
    bgClass: 'bg-yellow-500',
    textClass: 'text-yellow-500',
    borderClass: 'border-yellow-500',
    startPos: 26,
    kills: 0,
    sixes: 0
  },
  green: {
    id: 'green',
    name: 'Green',
    color: '#22c55e',
    darkColor: '#166534',
    bgClass: 'bg-green-500',
    textClass: 'text-green-500',
    borderClass: 'border-green-500',
    startPos: 39,
    kills: 0,
    sixes: 0
  },
};
