#!/usr/bin/env python3
"""
Neural Evaluator Server for Luddo AI Engine

Provides fast position evaluation using the trained MLX model.
Runs as a lightweight HTTP server on port 3021.
"""

import os
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import List, Dict, Any
import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

# Import model architecture
from train import LuddoEvaluator, load_model

# Constants
PLAYER_COLORS = ['red', 'blue', 'yellow', 'green']
PLAYER_START_POSITIONS = {'red': 0, 'blue': 13, 'yellow': 26, 'green': 39}
SAFE_SPOTS = [0, 8, 13, 21, 26, 34, 39, 47]
START_SPOTS = [0, 13, 26, 39]

# Global model
MODEL = None


def get_global_pos(player_color: str, step_count: int) -> int:
    """Calculate global position from step count"""
    if step_count < 0:
        return -1
    if step_count > 50:
        return -2
    if step_count >= 56:
        return 99
    return (PLAYER_START_POSITIONS[player_color] + step_count) % 52


def is_safe_spot(position: int) -> bool:
    return position in SAFE_SPOTS or position in START_SPOTS


def calculate_distance(from_pos: int, to_pos: int) -> int:
    if from_pos < 0 or to_pos < 0:
        return 100
    distance = (to_pos - from_pos + 52) % 52
    return distance if distance > 0 else 52


def get_threat_level(token: Dict, player_color: str, state: Dict) -> float:
    """Calculate threat level for a token"""
    if token['position'] < 0 or token['position'] == 99 or token['stepCount'] > 50:
        return 0.0

    if is_safe_spot(token['position']):
        return 0.0

    token_pos = get_global_pos(player_color, token['stepCount'])
    threat_count = 0
    min_distance = 7

    for opp_color in state['activeTurnOrder']:
        if opp_color == player_color:
            continue
        for opp_token in state['players'][opp_color]['tokens']:
            if opp_token['position'] < 0 or opp_token['stepCount'] > 50:
                continue
            opp_pos = get_global_pos(opp_color, opp_token['stepCount'])
            distance = calculate_distance(opp_pos, token_pos)
            if 1 <= distance <= 6:
                threat_count += 1
                min_distance = min(min_distance, distance)

    if threat_count == 0:
        return 0.0

    base_threat = (7 - min_distance) / 6
    multi_threat_bonus = min(threat_count - 1, 2) * 0.15
    return min(1.0, base_threat + multi_threat_bonus)


def extract_features(state: Dict, for_player: str) -> List[float]:
    """
    Extract 64 position features from game state.

    Feature vector (64 dims = 16 tokens × 4 features):
    For each token (4 per player × 4 players = 16 tokens):
    - Normalized position (0-1): progress towards finishing
    - Is at home/yard (0/1)
    - Is finished (0/1)
    - Distance to finish (normalized 0-1)

    Players are rotated so perspective player is always first.
    This matches the format used in feature_extractor.py for training.
    """
    features = []

    # Order players starting from perspective player
    player_order = state['activeTurnOrder'].copy()
    while player_order[0] != for_player:
        player_order.append(player_order.pop(0))

    # Pad to 4 players
    while len(player_order) < 4:
        player_order.append(player_order[0])

    for color in player_order:
        player = state['players'][color]
        for token in player['tokens']:
            step = token.get('stepCount', -1)
            pos = token.get('position', -1)

            # Determine token state
            is_home = (pos == -1)
            is_finished = (pos == 99 or step >= 56)

            if is_home:
                # At home/yard: [0, 1, 0, 1]
                features.extend([0.0, 1.0, 0.0, 1.0])
            elif is_finished:
                # Finished: [1, 0, 1, 0]
                features.extend([1.0, 0.0, 1.0, 0.0])
            else:
                # On board: calculate normalized position and distance
                # Use 57.0 scale to match training data (Python game uses 0-57)
                # TypeScript stepCount 0-56 maps to Python position 0-57
                pos_normalized = max(0, step) / 57.0
                distance_to_finish = max(0, 57 - step) / 57.0
                features.extend([pos_normalized, 0.0, 0.0, distance_to_finish])

    return features


def evaluate_position(state: Dict, player_color: str) -> float:
    """Evaluate position using neural network"""
    global MODEL

    if MODEL is None:
        return 0.5  # Fallback

    features = extract_features(state, player_color)
    features_mx = mx.array([features], dtype=mx.float32)

    prediction = MODEL(features_mx)
    return float(prediction[0, 0])


class NeuralHandler(BaseHTTPRequestHandler):
    """HTTP request handler for neural evaluations"""

    def do_POST(self):
        if self.path == '/evaluate':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)

            try:
                data = json.loads(post_data.decode('utf-8'))
                state = data['gameState']
                player = data['playerColor']

                score = evaluate_position(state, player)

                response = {'success': True, 'score': score}
                self.send_response(200)

            except Exception as e:
                response = {'success': False, 'error': str(e)}
                self.send_response(500)

            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))

        elif self.path == '/batch_evaluate':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)

            try:
                data = json.loads(post_data.decode('utf-8'))
                states = data['states']  # List of {gameState, playerColor}

                scores = []
                for item in states:
                    score = evaluate_position(item['gameState'], item['playerColor'])
                    scores.append(score)

                response = {'success': True, 'scores': scores}
                self.send_response(200)

            except Exception as e:
                response = {'success': False, 'error': str(e)}
                self.send_response(500)

            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
        elif self.path == '/evaluate_features':
            # Direct feature evaluation for MCTS
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)

            try:
                data = json.loads(post_data.decode('utf-8'))
                features = data['features']  # 64-dim feature vector
                if MODEL is None:
                    score = 0.5
                else:
                    features_mx = mx.array([features], dtype=mx.float32)
                    prediction = MODEL(features_mx)
                    score = float(prediction[0, 0])

                response = {'success': True, 'score': score}
                self.send_response(200)

            except Exception as e:
                response = {'success': False, 'error': str(e)}
                self.send_response(500)

            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))

        elif self.path == '/batch_evaluate_features':
            # Batch feature evaluation for MCTS
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)

            try:
                data = json.loads(post_data.decode('utf-8'))
                feature_list = data['features']  # List of 64-dim feature vectors

                scores = []
                if MODEL is None:
                    scores = [0.5] * len(feature_list)
                else:
                    features_mx = mx.array(feature_list, dtype=mx.float32)
                    predictions = MODEL(features_mx)
                    scores = [float(p[0]) for p in predictions]

                response = {'success': True, 'scores': scores}
                self.send_response(200)

            except Exception as e:
                response = {'success': False, 'error': str(e)}
                self.send_response(500)

            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))


        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == '/health':
            response = {
                'status': 'healthy',
                'model_loaded': MODEL is not None,
                'mlx_available': HAS_MLX
            }
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress logging for cleaner output
        pass


def main():
    global MODEL

    if not HAS_MLX:
        print("Error: MLX not available. Install with: pip install mlx")
        return

    # Load model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.npz')

    if os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        MODEL = load_model(model_path)
        print("Model loaded successfully!")
    else:
        print(f"Warning: Model not found at {model_path}")
        print("Running without neural evaluation (will return 0.5 for all positions)")

    # Start server
    port = 3021
    server = HTTPServer(('0.0.0.0', port), NeuralHandler)

    print(f"\n╔═══════════════════════════════════════════════╗")
    print(f"║     Neural Evaluator Server - Port {port}       ║")
    print(f"╠═══════════════════════════════════════════════╣")
    print(f"║  POST /evaluate       - Single evaluation     ║")
    print(f"║  POST /batch_evaluate - Batch evaluation      ║")
    print(f"║  GET  /health         - Health check          ║")
    print(f"╚═══════════════════════════════════════════════╝\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
