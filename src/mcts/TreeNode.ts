/**
 * MCTS Tree Node
 * Represents a node in the Monte Carlo search tree
 */

import type { OnlineGameState, PlayerColor } from '../types/types.js';

export class TreeNode {
  state: OnlineGameState;
  parent: TreeNode | null;
  children: TreeNode[];
  move: number | null;        // tokenId that led to this state
  diceValue: number | null;   // dice value for this move
  visits: number;
  wins: number;
  isChanceNode: boolean;      // Dice roll node (expectiminimax)
  untriedMoves: number[];
  player: PlayerColor;        // Player who made this move

  constructor(
    state: OnlineGameState,
    parent: TreeNode | null = null,
    move: number | null = null,
    diceValue: number | null = null,
    isChanceNode: boolean = false
  ) {
    this.state = state;
    this.parent = parent;
    this.children = [];
    this.move = move;
    this.diceValue = diceValue;
    this.visits = 0;
    this.wins = 0;
    this.isChanceNode = isChanceNode;
    this.untriedMoves = [];
    this.player = state.currentTurn;
  }

  /**
   * UCT (Upper Confidence Bound for Trees) value
   */
  uctValue(explorationConstant: number = Math.sqrt(2)): number {
    if (this.visits === 0) return Infinity;
    if (!this.parent) return this.wins / this.visits;

    const exploitation = this.wins / this.visits;
    const exploration = explorationConstant * Math.sqrt(Math.log(this.parent.visits) / this.visits);

    return exploitation + exploration;
  }

  /**
   * Check if node is fully expanded
   */
  isFullyExpanded(): boolean {
    return this.untriedMoves.length === 0;
  }

  /**
   * Check if node is a terminal state
   */
  isTerminal(): boolean {
    return this.state.gameState === 'finished';
  }

  /**
   * Get win rate for this node
   */
  winRate(): number {
    return this.visits > 0 ? this.wins / this.visits : 0;
  }

  /**
   * Select best child using UCT
   */
  selectBestChild(explorationConstant: number = Math.sqrt(2)): TreeNode | null {
    if (this.children.length === 0) return null;

    let bestChild: TreeNode | null = null;
    let bestUct = -Infinity;

    for (const child of this.children) {
      const uct = child.uctValue(explorationConstant);
      if (uct > bestUct) {
        bestUct = uct;
        bestChild = child;
      }
    }

    return bestChild;
  }

  /**
   * Select most visited child (for final move selection)
   */
  selectMostVisitedChild(): TreeNode | null {
    if (this.children.length === 0) return null;

    let bestChild: TreeNode | null = null;
    let bestVisits = -1;

    for (const child of this.children) {
      if (child.visits > bestVisits) {
        bestVisits = child.visits;
        bestChild = child;
      }
    }

    return bestChild;
  }

  /**
   * Add a child node
   */
  addChild(state: OnlineGameState, move: number, diceValue: number): TreeNode {
    const child = new TreeNode(state, this, move, diceValue);
    this.children.push(child);

    // Remove from untried moves
    const idx = this.untriedMoves.indexOf(move);
    if (idx > -1) {
      this.untriedMoves.splice(idx, 1);
    }

    return child;
  }

  /**
   * Backpropagate result up the tree
   */
  backpropagate(winner: PlayerColor | null): void {
    this.visits += 1;

    if (winner === this.player) {
      this.wins += 1;
    } else if (winner === null) {
      // Draw or ongoing - partial credit
      this.wins += 0.5;
    }
    // Loss: wins += 0

    if (this.parent) {
      this.parent.backpropagate(winner);
    }
  }

  /**
   * Get depth of this node
   */
  depth(): number {
    let depth = 0;
    let node: TreeNode | null = this;
    while (node.parent) {
      depth++;
      node = node.parent;
    }
    return depth;
  }

  /**
   * Get statistics string for debugging
   */
  toString(): string {
    return `TreeNode(move=${this.move}, visits=${this.visits}, wins=${this.wins}, winRate=${this.winRate().toFixed(2)})`;
  }
}
