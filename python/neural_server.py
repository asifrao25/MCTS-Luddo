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
    """Extract 64 position features from game state"""
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
            # Feature 1: Progress
            if token['position'] == 99:
                progress = 1.0
            elif token['stepCount'] >= 0:
                progress = token['stepCount'] / 56.0
            else:
                progress = 0.0

            # Feature 2: In yard
            in_yard = 1.0 if token['position'] == -1 else 0.0

            # Feature 3: In home stretch
            in_home_stretch = 1.0 if 50 < token['stepCount'] < 56 else 0.0

            # Feature 4: Threat level
            threat = get_threat_level(token, color, state)

            features.extend([progress, in_yard, in_home_stretch, threat])

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
    server = HTTPServer(('localhost', port), NeuralHandler)

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
