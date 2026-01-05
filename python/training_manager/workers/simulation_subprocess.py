#!/usr/bin/env python3
"""
Standalone simulation runner - runs as a separate process.
Communicates progress via a JSON file.
"""

import sys
import json
import time
import os
import traceback
import logging
from datetime import datetime
from pathlib import Path

# Configure logging to stderr for PM2 to capture
logging.basicConfig(
    level=logging.INFO,
    format='[SIM] %(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from game.game_engine import LudoGame
from game.feature_extractor import extract_all_player_features
from config import DATA_DIR


def run_simulation(run_id: str, config: dict, progress_file: str):
    """Run simulation and write progress to file."""
    num_players = config.get('numPlayers', 4)
    target_games = config.get('targetGames', 100)
    output_dir = config.get('outputDir', f'sim_{run_id[:8]}')
    mcts_simulations = config.get('mctsSimulations', 20)
    use_neural = config.get('useNeural', False)

    logger.info(f"Starting simulation: {run_id[:8]}")
    logger.info(f"Config: players={num_players}, games={target_games}, mcts={mcts_simulations}, neural={use_neural}")

    # Test neural server if enabled
    if use_neural:
        try:
            import httpx
            client = httpx.Client(timeout=5.0)
            test_features = [0.5] * 64
            resp = client.post("http://localhost:3021/evaluate_features", json={"features": test_features}, timeout=2.0)
            if resp.status_code == 200:
                logger.info(f"Neural server test OK: {resp.json()}")
            else:
                logger.warning(f"Neural server test failed: HTTP {resp.status_code}")
        except Exception as e:
            logger.error(f"Neural server test error: {e}")

    output_path = DATA_DIR / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    games_completed = 0
    positions_generated = 0
    all_positions = []
    start_time = time.time()

    # Write initial progress
    write_progress(progress_file, {
        'status': 'running',
        'gamesCompleted': 0,
        'positionsGenerated': 0,
        'gamesPerSecond': 0,
        'currentTurn': 0,
        'currentPlayer': 0
    })

    for game_num in range(target_games):
        # Check for cancellation
        if os.path.exists(f'{progress_file}.cancel'):
            logger.info("Cancellation requested")
            write_progress(progress_file, {
                'status': 'cancelled',
                'gamesCompleted': games_completed,
                'positionsGenerated': positions_generated
            })
            return

        try:
            game = LudoGame(num_players=num_players)
            game_positions = []
            turn_count = 0

            while not game.is_game_over():
                turn_count += 1

                features = extract_all_player_features(game)
                game_positions.append({
                    'features': features,
                    'turn': turn_count
                })

                game.make_mcts_move(
                    num_simulations=mcts_simulations,
                    use_neural=use_neural
                )

                # Update progress every 10 turns
                if turn_count % 10 == 0:
                    elapsed = time.time() - start_time
                    gps = games_completed / elapsed if elapsed > 0 else 0
                    write_progress(progress_file, {
                        'status': 'running',
                        'gamesCompleted': games_completed,
                        'positionsGenerated': positions_generated,
                        'gamesPerSecond': round(gps, 4),
                        'currentTurn': turn_count,
                        'currentPlayer': game.current_player,
                        'currentGame': game_num + 1
                    })

            winner = game.get_winner()
            for pos in game_positions:
                pos['winner'] = winner

            all_positions.extend(game_positions)
            games_completed += 1
            positions_generated += len(game_positions) * num_players

            if games_completed % 10 == 0:
                logger.info(f"Completed game {games_completed}/{target_games}, {positions_generated} positions")

            # Update progress after each game
            elapsed = time.time() - start_time
            gps = games_completed / elapsed if elapsed > 0 else 0
            write_progress(progress_file, {
                'status': 'running',
                'gamesCompleted': games_completed,
                'positionsGenerated': positions_generated,
                'gamesPerSecond': round(gps, 4),
                'currentTurn': turn_count,
                'currentPlayer': 0,
                'currentGame': game_num + 1
            })

        except Exception as e:
            logger.error(f"Error in game {game_num + 1}: {e}")
            logger.error(traceback.format_exc())
            raise

    # Save data
    positions_file = output_path / f'positions_{run_id[:8]}.json'
    with open(positions_file, 'w') as f:
        json.dump(all_positions, f)

    metadata_file = output_path / f'metadata_{run_id[:8]}.json'
    with open(metadata_file, 'w') as f:
        json.dump({
            'run_id': run_id,
            'num_games': games_completed,
            'num_positions': positions_generated,
            'num_players': num_players,
            'mcts_simulations': mcts_simulations,
            'use_neural': use_neural,
            'timestamp': datetime.utcnow().isoformat()
        }, f)

    logger.info(f"Simulation complete: {games_completed} games, {positions_generated} positions")
    logger.info(f"Data saved to: {output_path}")

    # Final progress
    write_progress(progress_file, {
        'status': 'completed',
        'gamesCompleted': games_completed,
        'positionsGenerated': positions_generated,
        'dataPath': str(output_path)
    })


def write_progress(filepath: str, data: dict):
    """Atomically write progress to file."""
    tmp = filepath + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f)
    os.rename(tmp, filepath)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: simulation_subprocess.py <run_id> <config_json> <progress_file>')
        sys.exit(1)
    
    run_id = sys.argv[1]
    config = json.loads(sys.argv[2])
    progress_file = sys.argv[3]
    
    try:
        run_simulation(run_id, config, progress_file)
    except Exception as e:
        write_progress(progress_file, {
            'status': 'failed',
            'error': str(e)
        })
        raise
