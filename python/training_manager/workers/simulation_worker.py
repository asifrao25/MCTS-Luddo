"""
Simulation worker - spawns subprocess for CPU-bound work.
"""

import json
import asyncio
import subprocess
import os
from datetime import datetime
from typing import Dict, Any, Set
from pathlib import Path

from ..database.connection import SessionLocal
from ..database.models import SimulationRunDB
from ..services.sse_manager import sse_manager
from ..config import DATA_DIR

# Track processes
_running_processes: Dict[str, subprocess.Popen] = {}
_cancelled_runs: Set[str] = set()


def cancel_simulation(run_id: str):
    """Signal a simulation to stop."""
    _cancelled_runs.add(run_id)
    # Create cancel file
    progress_file = f'/tmp/sim_{run_id}.json'
    Path(f'{progress_file}.cancel').touch()


def is_cancelled(run_id: str) -> bool:
    return run_id in _cancelled_runs


async def run_simulation(run_id: str, config: Dict[str, Any]):
    """
    Run simulation via subprocess - keeps server responsive.
    """
    print(f"[SIMULATION] Starting run_id={run_id}", flush=True)
    db = SessionLocal()
    log_handle = None

    try:
        run = db.query(SimulationRunDB).filter(SimulationRunDB.id == run_id).first()
        if not run:
            print(f"[SIMULATION] ERROR: Run {run_id} not found", flush=True)
            return

        progress_file = f'/tmp/sim_{run_id}.json'
        
        # Remove any old cancel file
        cancel_file = f'{progress_file}.cancel'
        if os.path.exists(cancel_file):
            os.remove(cancel_file)

        # Start subprocess
        script_path = Path(__file__).parent / 'simulation_subprocess.py'
        config_json = json.dumps(config)

        # Log file for subprocess output (prevents pipe buffer blocking!)
        log_file = f'/tmp/sim_{run_id}.log'
        log_handle = open(log_file, 'w')

        process = subprocess.Popen(
            ['/opt/homebrew/bin/python3', '-u', str(script_path), run_id, config_json, progress_file],
            stdout=log_handle,
            stderr=log_handle
        )
        _running_processes[run_id] = process

        print(f"[SIMULATION] Subprocess started PID={process.pid}, log={log_file}", flush=True)

        last_progress = {}
        last_game = 0

        # Poll for progress
        while True:
            await asyncio.sleep(0.5)  # Check every 500ms - doesn't block event loop!

            # Check if cancelled
            if is_cancelled(run_id):
                Path(cancel_file).touch()

            # Check if process finished
            if process.poll() is not None:
                break

            # Read progress file
            if os.path.exists(progress_file):
                try:
                    with open(progress_file, 'r') as f:
                        progress = json.load(f)
                    
                    if progress != last_progress:
                        # Update database
                        run.games_completed = progress.get('gamesCompleted', 0)
                        run.positions_generated = progress.get('positionsGenerated', 0)
                        db.commit()

                        # Emit SSE events
                        current_game = progress.get('currentGame', 0)
                        if current_game > last_game:
                            await sse_manager.emit(run_id, 'game_complete', {
                                'gameNumber': current_game,
                                'totalTurns': progress.get('currentTurn', 0)
                            })
                            last_game = current_game

                        await sse_manager.emit(run_id, 'progress', {
                            'gamesCompleted': progress.get('gamesCompleted', 0),
                            'positionsGenerated': progress.get('positionsGenerated', 0),
                            'gamesPerSecond': progress.get('gamesPerSecond', 0),
                            'currentTurn': progress.get('currentTurn', 0),
                            'currentPlayer': progress.get('currentPlayer', 0)
                        })

                        last_progress = progress.copy()

                except (json.JSONDecodeError, FileNotFoundError):
                    pass

        # Process finished - read final status
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                progress = json.load(f)
            
            status = progress.get('status', 'failed')
            run.status = status
            run.games_completed = progress.get('gamesCompleted', 0)
            run.positions_generated = progress.get('positionsGenerated', 0)
            run.data_path = progress.get('dataPath')
            run.end_time = datetime.utcnow()
            
            if status == 'failed':
                run.error_message = progress.get('error', 'Unknown error')
            
            db.commit()

            await sse_manager.emit(run_id, 'complete' if status == 'completed' else status, {
                'runId': run_id,
                'totalGames': run.games_completed,
                'totalPositions': run.positions_generated,
                'dataPath': run.data_path
            })

        # Cleanup temp files (log_handle closed in finally)
        if os.path.exists(progress_file):
            os.remove(progress_file)
        if os.path.exists(cancel_file):
            os.remove(cancel_file)

        print(f"[SIMULATION] Complete: {run.games_completed} games, status={run.status}", flush=True)

    except Exception as e:
        print(f"[SIMULATION] ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()

        run = db.query(SimulationRunDB).filter(SimulationRunDB.id == run_id).first()
        if run:
            run.status = 'failed'
            run.end_time = datetime.utcnow()
            run.error_message = str(e)
            db.commit()

        await sse_manager.emit(run_id, 'error', {'message': str(e)})

    finally:
        _cancelled_runs.discard(run_id)
        _running_processes.pop(run_id, None)

        # Close log file handle if opened
        if log_handle is not None:
            try:
                log_handle.close()
            except Exception:
                pass

        import training_manager.routers.simulation as sim_router
        if sim_router._running_simulation == run_id:
            sim_router._running_simulation = None

        db.close()
        print(f"[SIMULATION] Worker finished for {run_id}", flush=True)
