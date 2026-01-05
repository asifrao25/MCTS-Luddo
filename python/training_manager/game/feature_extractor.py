"""
Feature extraction for neural network input.

Extracts features from ALL player perspectives at each position.
This is critical for proper training - we need data from every player's POV.
"""

import numpy as np
from typing import List, Dict, Any
from .game_engine import LudoGame, GameState


def extract_features(game: LudoGame, perspective_player: int) -> List[float]:
    """
    Extract features from a single player's perspective.

    Feature vector (64 dimensions = 16 tokens × 4 features):
    For each token (4 per player × 4 players = 16 tokens):
    - Normalized position (0-1)
    - Is at home (0/1)
    - Is finished (0/1)
    - Distance to finish (normalized)

    Players are rotated so perspective_player is always player 0.
    """
    features = []

    # Rotate player order so perspective_player sees themselves as player 0
    num_players = game.num_players

    for player_offset in range(4):  # Always 4 "slots" even if fewer players
        actual_player = (perspective_player + player_offset) % num_players

        if player_offset >= num_players:
            # Inactive player slot - fill with zeros
            features.extend([0.0] * 16)  # 4 tokens × 4 features
            continue

        tokens = game.get_player_tokens(actual_player)

        for token in tokens:
            if token.is_home:
                # At home
                features.extend([0.0, 1.0, 0.0, 1.0])  # pos, home, finished, dist
            elif token.is_finished:
                # Finished
                features.extend([1.0, 0.0, 1.0, 0.0])  # pos, home, finished, dist
            else:
                # On board
                pos_normalized = token.position / 57.0
                distance_to_finish = max(0, 57 - token.position) / 57.0
                features.extend([pos_normalized, 0.0, 0.0, distance_to_finish])

    return features


def extract_all_player_features(game: LudoGame) -> Dict[int, List[float]]:
    """
    Extract features from ALL active players' perspectives.

    This is critical for training - we need to see the game from every
    player's point of view to learn proper position evaluation.

    Returns:
        Dictionary mapping player index to their feature vector
    """
    all_features = {}

    for player in range(game.num_players):
        all_features[player] = extract_features(game, player)

    return all_features


def state_to_tensor(features: List[float]) -> np.ndarray:
    """Convert feature list to numpy array for model input."""
    return np.array(features, dtype=np.float32)


def batch_states_to_tensor(feature_list: List[List[float]]) -> np.ndarray:
    """Convert multiple feature vectors to batch tensor."""
    return np.array(feature_list, dtype=np.float32)


def extract_position_with_outcome(
    game: LudoGame,
    winner: int
) -> List[Dict[str, Any]]:
    """
    Extract training positions from ALL players' perspectives.

    For each player, records:
    - features: 64-dim feature vector from their perspective
    - outcome: 1.0 if they won, 0.0 if they lost

    This ensures balanced training data from all viewpoints.
    """
    positions = []

    all_features = extract_all_player_features(game)

    for player, features in all_features.items():
        positions.append({
            "features": features,
            "player": player,
            "outcome": 1.0 if player == winner else 0.0
        })

    return positions
