"""
Benchmark worker for model comparison.

Runs head-to-head matches between two models.
"""

import json
from datetime import datetime
from typing import Dict, Any, Set, List

from ..database.connection import SessionLocal
from ..database.models import BenchmarkRunDB
from ..services.sse_manager import sse_manager

# Track cancellation requests
_cancelled_runs: Set[str] = set()


def cancel_benchmark(run_id: str):
    """Signal benchmark to stop."""
    _cancelled_runs.add(run_id)


def is_cancelled(run_id: str) -> bool:
    """Check if benchmark should stop."""
    return run_id in _cancelled_runs


async def run_benchmark(
    run_id: str,
    model_a_path: str,
    model_b_path: str,
    config: Dict[str, Any]
):
    """
    Run benchmark in background.

    Plays games between two models and tracks statistics.
    Models alternate colors each game for fairness.
    """
    db = SessionLocal()

    try:
        run = db.query(BenchmarkRunDB).filter(BenchmarkRunDB.id == run_id).first()
        if not run:
            return

        num_players = config.get("numPlayers", 2)
        target_games = config.get("targetGames", 100)

        # Load models
        # model_a = load_model(model_a_path)
        # model_b = load_model(model_b_path)

        model_a_wins = 0
        model_b_wins = 0
        draws = 0
        game_details: List[Dict] = []

        # Import game engine
        from ..game.game_engine import LudoGame

        for game_num in range(target_games):
            if is_cancelled(run_id):
                break

            # Emit game start
            await sse_manager.emit(run_id, "game_start", {
                "gameNumber": game_num + 1,
                "gamesTotal": target_games
            })

            # Alternate which model plays first for fairness
            model_a_is_player_0 = (game_num % 2 == 0)

            # Create and play game
            game = LudoGame(num_players=num_players)
            turn_count = 0

            while not game.is_game_over():
                if is_cancelled(run_id):
                    break

                turn_count += 1

                # Determine which model to use
                current_player = game.current_player
                is_model_a_turn = (
                    (model_a_is_player_0 and current_player == 0) or
                    (not model_a_is_player_0 and current_player == 1)
                )

                # Make move using appropriate model
                if is_model_a_turn:
                    # move = model_a.get_best_move(game)
                    pass
                else:
                    # move = model_b.get_best_move(game)
                    pass

                game.make_best_move()  # Stub - use model move

                # Emit turn progress every 10 turns
                if turn_count % 10 == 0:
                    await sse_manager.emit(run_id, "game_progress", {
                        "gameNumber": game_num + 1,
                        "turnNumber": turn_count
                    })

            # Determine winner
            winner = game.get_winner()
            winner_model = None

            if winner is not None:
                if (model_a_is_player_0 and winner == 0) or \
                   (not model_a_is_player_0 and winner == 1):
                    model_a_wins += 1
                    winner_model = "A"
                else:
                    model_b_wins += 1
                    winner_model = "B"
            else:
                draws += 1

            # Record game details
            game_details.append({
                "gameNumber": game_num + 1,
                "modelAFirst": model_a_is_player_0,
                "winner": winner_model,
                "turns": turn_count
            })

            # Update database
            run.games_completed = game_num + 1
            run.model_a_wins = model_a_wins
            run.model_b_wins = model_b_wins
            run.draws = draws
            db.commit()

            # Emit game complete
            await sse_manager.emit(run_id, "game_complete", {
                "gameNumber": game_num + 1,
                "winnerModel": winner_model,
                "modelAWins": model_a_wins,
                "modelBWins": model_b_wins
            })

        # Calculate final statistics
        total_decisive = model_a_wins + model_b_wins
        win_rate_a = (model_a_wins / total_decisive * 100) if total_decisive > 0 else 50.0
        win_rate_b = (model_b_wins / total_decisive * 100) if total_decisive > 0 else 50.0

        # Determine overall winner
        if model_a_wins > model_b_wins:
            overall_winner = run.model_a_name
        elif model_b_wins > model_a_wins:
            overall_winner = run.model_b_name
        else:
            overall_winner = "Draw"

        # Update final status
        run.status = "cancelled" if is_cancelled(run_id) else "completed"
        run.end_time = datetime.utcnow()
        run.win_rate_a = win_rate_a
        run.win_rate_b = win_rate_b
        run.winner = overall_winner
        run.game_details = json.dumps(game_details)
        db.commit()

        # Emit completion
        await sse_manager.emit(run_id, "complete", {
            "runId": run_id,
            "modelAWins": model_a_wins,
            "modelBWins": model_b_wins,
            "winner": overall_winner,
            "winRateA": win_rate_a,
            "winRateB": win_rate_b
        })

    except Exception as e:
        # Update error status
        run = db.query(BenchmarkRunDB).filter(BenchmarkRunDB.id == run_id).first()
        if run:
            run.status = "failed"
            run.end_time = datetime.utcnow()
            run.error_message = str(e)
            db.commit()

        await sse_manager.emit(run_id, "error", {"message": str(e)})

    finally:
        _cancelled_runs.discard(run_id)

        # Clear running benchmark
        import training_manager.routers.benchmark as bench_router
        if bench_router._running_benchmark == run_id:
            bench_router._running_benchmark = None

        db.close()
