"""
Ludo game engine for simulation and benchmarking.

Supports 2-4 players with standard Ludo rules.
"""

import random
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class Token:
    """Represents a single token on the board."""
    player: int
    index: int  # 0-3 for each player's tokens
    position: int = -1  # -1 = home, 0-51 = board, 52-57 = home stretch
    is_home: bool = True
    is_finished: bool = False


@dataclass
class GameState:
    """Complete game state."""
    num_players: int
    tokens: List[Token] = field(default_factory=list)
    current_player: int = 0
    dice_value: int = 0
    turn_count: int = 0
    winner: Optional[int] = None
    active_turn_order: List[int] = field(default_factory=list)


class LudoGame:
    """
    Ludo game implementation.

    Board layout:
    - Positions 0-51: Main board (52 squares)
    - Positions 52-57: Home stretch (6 squares per player)
    - Position -1: Token at home
    - Position 58+: Token finished
    """

    # Starting positions for each player on the main board
    START_POSITIONS = {0: 0, 1: 13, 2: 26, 3: 39}

    # Entry to home stretch for each player
    HOME_ENTRY = {0: 50, 1: 11, 2: 24, 3: 37}

    # All safe spots (start positions + star squares)
    SAFE_SPOTS = {0, 8, 13, 21, 26, 34, 39, 47}

    def __init__(self, num_players: int = 2):
        """Initialize game with specified number of players."""
        if num_players < 2 or num_players > 4:
            raise ValueError("Number of players must be 2-4")

        self.num_players = num_players
        self.tokens: List[Token] = []
        self.current_player = 0
        self.dice_value = 0
        self.turn_count = 0
        self.winner: Optional[int] = None
        self.last_captured = False  # Track if last move captured an opponent
        self.last_reached_home = False  # Track if last move reached home

        # Create tokens for each player
        for player in range(num_players):
            for i in range(4):
                self.tokens.append(Token(player=player, index=i))

        # Active players in turn order
        self.active_turn_order = list(range(num_players))

    def get_state(self) -> GameState:
        """Get current game state."""
        return GameState(
            num_players=self.num_players,
            tokens=self.tokens.copy(),
            current_player=self.current_player,
            dice_value=self.dice_value,
            turn_count=self.turn_count,
            winner=self.winner,
            active_turn_order=self.active_turn_order.copy()
        )

    def roll_dice(self) -> int:
        """Roll dice and return value."""
        self.dice_value = random.randint(1, 6)
        return self.dice_value

    def get_player_tokens(self, player: int) -> List[Token]:
        """Get all tokens for a player."""
        return [t for t in self.tokens if t.player == player]

    def get_valid_moves(self) -> List[Dict[str, Any]]:
        """Get all valid moves for current player with current dice."""
        moves = []
        player_tokens = self.get_player_tokens(self.current_player)

        for token in player_tokens:
            if token.is_finished:
                continue

            if token.is_home:
                # Can only leave home on 6
                if self.dice_value == 6:
                    moves.append({
                        "token_index": token.index,
                        "from_pos": -1,
                        "to_pos": self.START_POSITIONS[self.current_player],
                        "type": "leave_home"
                    })
            else:
                # Calculate new position
                new_pos = self._calculate_new_position(token, self.dice_value)
                if new_pos is not None:
                    move_type = "finish" if new_pos == 57 else "move"
                    moves.append({
                        "token_index": token.index,
                        "from_pos": token.position,
                        "to_pos": new_pos,
                        "type": move_type
                    })

        return moves

    def _calculate_new_position(self, token: Token, steps: int) -> Optional[int]:
        """Calculate new position for token after moving steps."""
        current_pos = token.position
        player = token.player

        # Already in home stretch
        if current_pos >= 52:
            new_pos = current_pos + steps
            if new_pos > 57:
                return None  # Would overshoot
            return new_pos if new_pos <= 57 else None

        # On main board
        home_entry = self.HOME_ENTRY[player]
        new_pos = current_pos

        for _ in range(steps):
            if new_pos == home_entry:
                # Enter home stretch
                new_pos = 52
            elif new_pos >= 52:
                new_pos += 1
            else:
                new_pos = (new_pos + 1) % 52

        if new_pos > 57:
            return None

        return new_pos

    def make_move(self, move: Dict[str, Any]) -> bool:
        """Execute a move.

        Also sets self.last_captured and self.last_reached_home for extra turn logic.
        """
        token_index = move["token_index"]
        token = None
        for t in self.tokens:
            if t.player == self.current_player and t.index == token_index:
                token = t
                break

        if token is None:
            return False

        # Reset extra turn flags
        self.last_captured = False
        self.last_reached_home = False

        # Update token position
        token.position = move["to_pos"]
        token.is_home = False

        if move["type"] == "finish" or move["to_pos"] == 57:
            token.is_finished = True
            token.position = 58
            self.last_reached_home = True

        # Check for captures
        if move["type"] not in ["leave_home", "finish"] and move["to_pos"] < 52:
            self.last_captured = self._check_captures(move["to_pos"])

        return True

    def _check_captures(self, position: int) -> bool:
        """Check if any opponent tokens are captured at position.

        Returns True if a capture occurred.
        """
        # Safe squares (can't capture) - includes start positions and star squares
        if position in self.SAFE_SPOTS:
            return False

        captured = False
        for token in self.tokens:
            if token.player != self.current_player and \
               not token.is_home and \
               not token.is_finished and \
               token.position == position:
                # Capture - send back home
                token.position = -1
                token.is_home = True
                captured = True

        return captured

    def next_turn(self):
        """Advance to next player's turn.

        Player gets another turn if:
        - Rolled a 6
        - Captured an opponent's token
        - A token reached home (finished)
        """
        self.turn_count += 1

        # Extra turn conditions: roll 6, capture, or reach home
        gets_extra_turn = (
            self.dice_value == 6 or
            self.last_captured or
            self.last_reached_home
        )

        if not gets_extra_turn:
            self.current_player = (self.current_player + 1) % self.num_players
            # Skip eliminated players
            while self.current_player not in self.active_turn_order:
                self.current_player = (self.current_player + 1) % self.num_players

        # Reset flags for next move
        self.last_captured = False
        self.last_reached_home = False

    def is_game_over(self) -> bool:
        """Check if game is over."""
        if self.winner is not None:
            return True

        for player in range(self.num_players):
            tokens = self.get_player_tokens(player)
            if all(t.is_finished for t in tokens):
                self.winner = player
                return True

        return False

    def get_winner(self) -> Optional[int]:
        """Get winning player or None."""
        return self.winner

    def make_best_move(self):
        """Make best move using simple heuristic."""
        self.roll_dice()
        moves = self.get_valid_moves()

        if not moves:
            self.next_turn()
            return

        # Simple heuristic: prioritize finishing, then captures, then advancing
        best_move = None
        best_score = -float('inf')

        for move in moves:
            score = 0

            if move["type"] == "finish":
                score = 1000
            elif move["type"] == "leave_home" and self.dice_value == 6:
                score = 100
            else:
                # Prefer advancing tokens that are further along
                score = move["to_pos"]

                # Check for potential capture
                for token in self.tokens:
                    if token.player != self.current_player and \
                       not token.is_home and \
                       token.position == move["to_pos"]:
                        score += 200

            if score > best_score:
                best_score = score
                best_move = move

        if best_move:
            self.make_move(best_move)

        self.next_turn()

    def clone(self) -> 'LudoGame':
        """Create a deep copy of the game."""
        new_game = LudoGame(self.num_players)
        new_game.tokens = [
            Token(
                player=t.player,
                index=t.index,
                position=t.position,
                is_home=t.is_home,
                is_finished=t.is_finished
            )
            for t in self.tokens
        ]
        new_game.current_player = self.current_player
        new_game.dice_value = self.dice_value
        new_game.turn_count = self.turn_count
        new_game.winner = self.winner
        new_game.active_turn_order = self.active_turn_order.copy()
        new_game.last_captured = self.last_captured
        new_game.last_reached_home = self.last_reached_home
        return new_game


    def make_mcts_move(self, num_simulations: int = 50, use_neural: bool = True):
        """Make move using MCTS with neural network evaluation."""
        from .mcts import MCTS
        
        mcts = MCTS(num_simulations=num_simulations, use_neural=use_neural)
        
        # Roll dice first
        self.roll_dice()
        
        # Get MCTS recommendation
        result = mcts.search(self)
        move = result.get('move')
        
        if move:
            self.make_move(move)
        
        self.next_turn()
        
        return result  # Return search statistics for analysis
