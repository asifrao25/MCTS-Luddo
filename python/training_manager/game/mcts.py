"""
Monte Carlo Tree Search (MCTS) for Luddo AI.

Implements AlphaZero-style MCTS with neural network evaluation.
Uses UCB1 for selection and the neural server for position evaluation.
"""

import math
import random
import sys
import httpx
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from .game_engine import LudoGame
from .feature_extractor import extract_features


# Neural server endpoint
NEURAL_SERVER_URL = "http://localhost:3021"


@dataclass
class MCTSNode:
    """Node in the MCTS tree."""
    game: LudoGame
    parent: Optional["MCTSNode"] = None
    move: Optional[Dict[str, Any]] = None
    children: Dict[str, "MCTSNode"] = field(default_factory=dict)
    
    visits: int = 0
    value_sum: float = 0.0
    prior: float = 1.0
    
    @property
    def value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits
    
    def ucb_score(self, c_puct: float = 1.4) -> float:
        if self.visits == 0:
            return float("inf")
        parent_visits = self.parent.visits if self.parent else 1
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visits)
        return self.value + exploration
    
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def is_terminal(self) -> bool:
        return self.game.is_game_over()


class MCTS:
    """
    Monte Carlo Tree Search with neural network evaluation.
    """
    
    def __init__(
        self,
        num_simulations: int = 50,
        c_puct: float = 1.4,
        temperature: float = 1.0,
        use_neural: bool = True
    ):
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.use_neural = use_neural
        self._http_client = None
    
    def get_http_client(self):
        if self._http_client is None:
            self._http_client = httpx.Client(timeout=5.0)
        return self._http_client
    
    def evaluate_position(self, game: LudoGame) -> float:
        """Evaluate position using neural network."""
        if not self.use_neural:
            return random.uniform(0.4, 0.6)

        try:
            features = extract_features(game, game.current_player)
            client = self.get_http_client()

            response = client.post(
                f"{NEURAL_SERVER_URL}/evaluate_features",
                json={"features": features},
                timeout=2.0
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("score", 0.5)
            print(f"[MCTS] Neural eval failed: HTTP {response.status_code}", file=sys.stderr)
            return 0.5

        except Exception as e:
            print(f"[MCTS] Neural eval error: {e}", file=sys.stderr)
            return 0.5
    
    def evaluate_with_rollout(self, game: LudoGame, max_moves: int = 100) -> float:
        """Evaluate using random rollout."""
        sim_game = game.clone()
        original_player = sim_game.current_player
        
        for _ in range(max_moves):
            if sim_game.is_game_over():
                break
            
            sim_game.roll_dice()
            moves = sim_game.get_valid_moves()
            if moves:
                move = random.choice(moves)
                sim_game.make_move(move)
            sim_game.next_turn()
        
        winner = sim_game.get_winner()
        return 1.0 if winner == original_player else 0.0
    
    def select(self, node: MCTSNode) -> MCTSNode:
        """Select leaf node using UCB."""
        while not node.is_leaf() and not node.is_terminal():
            best_score = -float("inf")
            best_child = None
            
            for child in node.children.values():
                score = child.ucb_score(self.c_puct)
                if score > best_score:
                    best_score = score
                    best_child = child
            
            if best_child is None:
                break
            node = best_child
        
        return node
    
    def expand(self, node: MCTSNode) -> None:
        """Expand node by adding all valid moves."""
        if node.is_terminal():
            return
        
        game = node.game
        game.roll_dice()
        moves = game.get_valid_moves()
        
        if not moves:
            child_game = game.clone()
            child_game.next_turn()
            child = MCTSNode(game=child_game, parent=node, move=None)
            node.children["pass"] = child
            return
        
        for move in moves:
            child_game = game.clone()
            child_game.dice_value = game.dice_value
            child_game.make_move(move)
            child_game.next_turn()
            
            move_key = f"{move['token_index']}_{move['to_pos']}"
            child = MCTSNode(game=child_game, parent=node, move=move)
            child.prior = 1.0 / len(moves)
            node.children[move_key] = child
    
    def backpropagate(self, node: MCTSNode, value: float) -> None:
        """Backpropagate value up the tree."""
        while node is not None:
            node.visits += 1
            node.value_sum += value
            value = 1.0 - value
            node = node.parent
    
    def search(self, game: LudoGame) -> Dict[str, Any]:
        """Run MCTS and return best move with statistics."""
        root = MCTSNode(game=game.clone())
        
        for _ in range(self.num_simulations):
            node = self.select(root)
            
            if not node.is_terminal() and node.visits > 0:
                self.expand(node)
                if node.children:
                    node = random.choice(list(node.children.values()))
            
            if node.is_terminal():
                winner = node.game.get_winner()
                root_player = root.game.current_player
                value = 1.0 if winner == root_player else 0.0
            else:
                if self.use_neural:
                    value = self.evaluate_position(node.game)
                    if node.game.current_player != root.game.current_player:
                        value = 1.0 - value
                else:
                    value = self.evaluate_with_rollout(node.game)
            
            self.backpropagate(node, value)
        
        if not root.children:
            return {"move": None, "visits": {}, "values": {}}
        
        visits = {k: c.visits for k, c in root.children.items()}
        values = {k: c.value for k, c in root.children.items()}
        
        if self.temperature == 0:
            best_key = max(root.children.keys(), key=lambda k: root.children[k].visits)
        else:
            total = sum(c.visits ** (1.0 / self.temperature) for c in root.children.values())
            if total == 0:
                best_key = random.choice(list(root.children.keys()))
            else:
                r = random.random()
                cumsum = 0.0
                best_key = list(root.children.keys())[-1]
                for k, c in root.children.items():
                    prob = (c.visits ** (1.0 / self.temperature)) / total
                    cumsum += prob
                    if r <= cumsum:
                        best_key = k
                        break
        
        best_move = root.children[best_key].move
        
        return {
            "move": best_move,
            "visits": visits,
            "values": values,
            "best_key": best_key
        }
    
    def get_best_move(self, game: LudoGame) -> Optional[Dict[str, Any]]:
        """Get just the best move."""
        result = self.search(game)
        return result["move"]


_default_mcts = None


def get_mcts(num_simulations: int = 50, use_neural: bool = True) -> MCTS:
    """Get or create MCTS instance."""
    global _default_mcts
    if _default_mcts is None or _default_mcts.num_simulations != num_simulations:
        _default_mcts = MCTS(num_simulations=num_simulations, use_neural=use_neural)
    return _default_mcts
