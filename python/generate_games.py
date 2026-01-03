#!/usr/bin/env python3
"""
Self-Play Game Generator for Luddo Neural Evaluator Training

Generates training data by having MCTS agents play against each other.
Extracts position features and game outcomes for supervised learning.
"""

import json
import random
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm
import numpy as np

# Game constants
PLAYER_COLORS = ['red', 'blue', 'yellow', 'green']
PLAYER_START_POSITIONS = {'red': 0, 'blue': 13, 'yellow': 26, 'green': 39}
SAFE_SPOTS = [0, 8, 13, 21, 26, 34, 39, 47]
START_SPOTS = [0, 13, 26, 39]


@dataclass
class Token:
    id: int
    color: str
    position: int  # -1=yard, 0-51=board, 99=home
    step_count: int  # -1=yard, 0-56=steps from start


@dataclass
class Player:
    id: str
    tokens: List[Token]
    start_pos: int
    active: bool = True


@dataclass
class GameState:
    players: Dict[str, Player]
    active_turn_order: List[str]
    current_turn: str
    dice_value: int
    winner: Optional[str] = None
    rankings: List[str] = None

    def __post_init__(self):
        if self.rankings is None:
            self.rankings = []


@dataclass
class PositionFeatures:
    """64 features: 4 players x 4 tokens x 4 features per token"""
    features: List[float]  # 64 floats
    outcome: float  # 1.0 = win, 0.5 = draw, 0.0 = loss
    player: str  # Player perspective


def get_global_pos(player_color: str, step_count: int) -> int:
    """Calculate global position from step count"""
    if step_count < 0:
        return -1
    if step_count > 50:
        return -2  # Home stretch
    if step_count >= 56:
        return 99
    start_pos = PLAYER_START_POSITIONS[player_color]
    return (start_pos + step_count) % 52


def is_safe_spot(position: int) -> bool:
    """Check if position is safe"""
    return position in SAFE_SPOTS or position in START_SPOTS


def calculate_distance(from_pos: int, to_pos: int) -> int:
    """Calculate circular board distance"""
    if from_pos < 0 or to_pos < 0:
        return 100  # Unreachable
    distance = (to_pos - from_pos + 52) % 52
    return distance if distance > 0 else 52


def get_threat_level(token: Token, player_color: str, state: GameState) -> float:
    """Calculate threat level (0-1) for a token"""
    if token.position < 0 or token.position == 99 or token.step_count > 50:
        return 0.0

    if is_safe_spot(token.position):
        return 0.0

    token_pos = get_global_pos(player_color, token.step_count)
    threat_count = 0
    min_distance = 7

    for opp_color in state.active_turn_order:
        if opp_color == player_color:
            continue
        for opp_token in state.players[opp_color].tokens:
            if opp_token.position < 0 or opp_token.step_count > 50:
                continue
            opp_pos = get_global_pos(opp_color, opp_token.step_count)
            distance = calculate_distance(opp_pos, token_pos)
            if 1 <= distance <= 6:
                threat_count += 1
                min_distance = min(min_distance, distance)

    if threat_count == 0:
        return 0.0

    # Higher threat for closer opponents and multiple threats
    base_threat = (7 - min_distance) / 6
    multi_threat_bonus = min(threat_count - 1, 2) * 0.15
    return min(1.0, base_threat + multi_threat_bonus)


def extract_features(state: GameState, for_player: str) -> List[float]:
    """Extract 64 position features for neural network input"""
    features = []

    # Order players starting from the perspective player
    player_order = state.active_turn_order.copy()
    while player_order[0] != for_player:
        player_order.append(player_order.pop(0))

    # Pad to 4 players
    while len(player_order) < 4:
        player_order.append(player_order[0])  # Duplicate for inactive slots

    for color in player_order:
        player = state.players[color]
        for token in player.tokens:
            # Feature 1: Progress (normalized 0-1)
            if token.position == 99:
                progress = 1.0
            elif token.step_count >= 0:
                progress = token.step_count / 56.0
            else:
                progress = 0.0

            # Feature 2: In yard (0/1)
            in_yard = 1.0 if token.position == -1 else 0.0

            # Feature 3: In home stretch (0/1)
            in_home_stretch = 1.0 if 50 < token.step_count < 56 else 0.0

            # Feature 4: Threat level (0-1)
            threat = get_threat_level(token, color, state)

            features.extend([progress, in_yard, in_home_stretch, threat])

    return features


def create_player(color: str) -> Player:
    """Create a new player with tokens in yard"""
    tokens = [
        Token(id=i, color=color, position=-1, step_count=-1)
        for i in range(4)
    ]
    return Player(
        id=color,
        tokens=tokens,
        start_pos=PLAYER_START_POSITIONS[color],
        active=True
    )


def create_game(num_players: int = 2) -> GameState:
    """Create a new game with specified number of players"""
    colors = PLAYER_COLORS[:num_players]
    players = {color: create_player(color) for color in colors}

    # Add inactive players
    for color in PLAYER_COLORS:
        if color not in players:
            players[color] = create_player(color)
            players[color].active = False

    return GameState(
        players=players,
        active_turn_order=colors,
        current_turn=colors[0],
        dice_value=1
    )


def get_valid_moves(player: Player, dice_value: int) -> List[int]:
    """Get valid token IDs for current dice value"""
    valid = []
    for token in player.tokens:
        if token.position == -1:
            if dice_value == 6:
                valid.append(token.id)
        elif token.position != 99:
            if token.step_count + dice_value <= 56:
                valid.append(token.id)
    return valid


def execute_move(state: GameState, token_id: int, dice_value: int) -> Tuple[bool, bool]:
    """Execute a move. Returns (captured, reached_home)"""
    player = state.players[state.current_turn]
    token = player.tokens[token_id]

    captured = False
    reached_home = False

    if token.position == -1:
        # Exit yard
        token.step_count = 0
        token.position = player.start_pos
    else:
        # Normal move
        token.step_count += dice_value

        if token.step_count == 56:
            token.position = 99
            reached_home = True
        elif token.step_count > 50:
            token.position = 100 + (token.step_count - 51)
        else:
            token.position = get_global_pos(state.current_turn, token.step_count)

    # Check captures
    if 0 <= token.position < 99 and token.position not in SAFE_SPOTS:
        for opp_color in state.active_turn_order:
            if opp_color == state.current_turn:
                continue
            for opp_token in state.players[opp_color].tokens:
                if opp_token.position == token.position:
                    opp_token.position = -1
                    opp_token.step_count = -1
                    captured = True

    return captured, reached_home


def check_winner(state: GameState) -> Optional[str]:
    """Check if current player has won"""
    player = state.players[state.current_turn]
    finished = sum(1 for t in player.tokens if t.position == 99)
    if finished == 4:
        return state.current_turn
    return None


def get_next_player(state: GameState) -> str:
    """Get next player in turn order"""
    current_idx = state.active_turn_order.index(state.current_turn)
    next_idx = (current_idx + 1) % len(state.active_turn_order)

    # Skip finished players
    attempts = 0
    while state.active_turn_order[next_idx] in state.rankings and attempts < len(state.active_turn_order):
        next_idx = (next_idx + 1) % len(state.active_turn_order)
        attempts += 1

    return state.active_turn_order[next_idx]


def evaluate_position(state: GameState, player_color: str) -> float:
    """Simple heuristic evaluation (0-1)"""
    player = state.players[player_color]
    score = 0.0

    for token in player.tokens:
        if token.position == 99:
            score += 100
        elif token.step_count > 50:
            score += 60 + (token.step_count - 50) * 5
        elif token.step_count >= 0:
            score += token.step_count
        else:
            score -= 10

    # Normalize to 0-1
    max_score = 400 + 30 * 4  # 4 tokens home + bonus
    return max(0.0, min(1.0, (score + 40) / (max_score + 40)))


def select_move_heuristic(state: GameState, valid_moves: List[int]) -> int:
    """Select move using simple heuristics"""
    if len(valid_moves) == 1:
        return valid_moves[0]

    player = state.players[state.current_turn]
    dice = state.dice_value

    best_move = valid_moves[0]
    best_score = -1000

    for token_id in valid_moves:
        token = player.tokens[token_id]
        score = 0

        # Prioritize reaching home
        if token.step_count >= 0 and token.step_count + dice == 56:
            score += 500

        # Prioritize exiting yard
        if token.position == -1 and dice == 6:
            score += 80

        # Prioritize entering home stretch
        if token.step_count <= 50 and token.step_count + dice > 50:
            score += 60

        # Prefer advancing furthest token
        if token.step_count >= 0:
            score += token.step_count

        if score > best_score:
            best_score = score
            best_move = token_id

    return best_move


def play_game(num_players: int = 2, max_moves: int = 500) -> Tuple[List[PositionFeatures], str]:
    """Play a complete game and collect training data"""
    state = create_game(num_players)
    positions: List[Tuple[GameState, str, List[float]]] = []
    move_count = 0

    while state.winner is None and move_count < max_moves:
        # Roll dice
        dice = random.randint(1, 6)
        state.dice_value = dice

        player = state.players[state.current_turn]
        valid_moves = get_valid_moves(player, dice)

        if not valid_moves:
            state.current_turn = get_next_player(state)
            move_count += 1
            continue

        # Record position before move
        features = extract_features(state, state.current_turn)
        positions.append((state.current_turn, features))

        # Select and execute move
        move = select_move_heuristic(state, valid_moves)
        captured, reached_home = execute_move(state, move, dice)

        # Check for win
        winner = check_winner(state)
        if winner:
            state.winner = winner
            state.rankings.append(winner)
            break

        # Extra turn for 6, capture, or reaching home
        if not (dice == 6 or captured or reached_home):
            state.current_turn = get_next_player(state)

        move_count += 1

    # Create training samples with outcomes
    training_data = []
    for player_color, features in positions:
        if state.winner:
            outcome = 1.0 if player_color == state.winner else 0.0
        else:
            outcome = 0.5  # Draw/timeout

        training_data.append(PositionFeatures(
            features=features,
            outcome=outcome,
            player=player_color
        ))

    return training_data, state.winner or "draw"


def generate_training_data(num_games: int, output_dir: str, num_players: int = 2):
    """Generate training data from self-play games"""
    os.makedirs(output_dir, exist_ok=True)

    all_features = []
    all_outcomes = []
    game_results = {'wins': {color: 0 for color in PLAYER_COLORS}, 'draws': 0}

    print(f"Generating {num_games} self-play games with {num_players} players...")

    for _ in tqdm(range(num_games)):
        positions, winner = play_game(num_players)

        if winner == "draw":
            game_results['draws'] += 1
        else:
            game_results['wins'][winner] += 1

        for pos in positions:
            all_features.append(pos.features)
            all_outcomes.append(pos.outcome)

    # Convert to numpy arrays
    features_array = np.array(all_features, dtype=np.float32)
    outcomes_array = np.array(all_outcomes, dtype=np.float32)

    # Save training data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    np.save(os.path.join(output_dir, f"features_{timestamp}.npy"), features_array)
    np.save(os.path.join(output_dir, f"outcomes_{timestamp}.npy"), outcomes_array)

    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'num_games': num_games,
        'num_players': num_players,
        'num_positions': len(all_features),
        'feature_dim': 64,
        'game_results': game_results
    }

    with open(os.path.join(output_dir, f"metadata_{timestamp}.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nGenerated {len(all_features)} training positions")
    print(f"Saved to: {output_dir}")
    print(f"Game results: {game_results}")

    return features_array, outcomes_array


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Luddo self-play training data")
    parser.add_argument("--games", type=int, default=10000, help="Number of games to generate")
    parser.add_argument("--players", type=int, default=2, help="Number of players per game")
    parser.add_argument("--output", type=str, default="../data/self_play_games", help="Output directory")

    args = parser.parse_args()

    generate_training_data(args.games, args.output, args.players)
