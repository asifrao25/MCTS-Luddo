"""
Simulation endpoints for training data generation.
"""

import uuid
import asyncio
import json
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from ..database.connection import get_db
from ..database.models import SimulationRunDB, SimulationRunCreate
from ..services.sse_manager import sse_manager, stream_generator
from ..config import DATA_DIR

router = APIRouter(prefix="/api/simulation", tags=["simulation"])

# Track running simulation
_running_simulation: Optional[str] = None


@router.get("/runs")
async def list_simulation_runs(
    limit: int = 20,
    offset: int = 0,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    List all simulation runs with pagination.
    """
    query = db.query(SimulationRunDB)

    if status:
        query = query.filter(SimulationRunDB.status == status)

    total = query.count()
    runs = query.order_by(
        SimulationRunDB.created_at.desc()
    ).offset(offset).limit(limit).all()

    return {
        "success": True,
        "data": {
            "runs": [
                {
                    "id": r.id,
                    "startTime": r.start_time.isoformat() if r.start_time else None,
                    "endTime": r.end_time.isoformat() if r.end_time else None,
                    "status": r.status,
                    "numPlayers": r.num_players,
                    "targetGames": r.target_games,
                    "gamesCompleted": r.games_completed,
                    "positionsGenerated": r.positions_generated,
                    "dataPath": r.data_path,
                    "createdAt": r.created_at.isoformat()
                }
                for r in runs
            ],
            "total": total,
            "hasMore": offset + limit < total
        }
    }


@router.get("/runs/{run_id}")
async def get_simulation_run(run_id: str, db: Session = Depends(get_db)):
    """
    Get details of a specific simulation run.
    """
    run = db.query(SimulationRunDB).filter(SimulationRunDB.id == run_id).first()

    if not run:
        raise HTTPException(status_code=404, detail="Simulation run not found")

    config = None
    if run.config:
        try:
            config = json.loads(run.config)
        except:
            pass

    return {
        "success": True,
        "data": {
            "id": run.id,
            "startTime": run.start_time.isoformat() if run.start_time else None,
            "endTime": run.end_time.isoformat() if run.end_time else None,
            "status": run.status,
            "numPlayers": run.num_players,
            "targetGames": run.target_games,
            "gamesCompleted": run.games_completed,
            "positionsGenerated": run.positions_generated,
            "dataPath": run.data_path,
            "config": config,
            "errorMessage": run.error_message,
            "createdAt": run.created_at.isoformat()
        }
    }


@router.post("/start")
async def start_simulation(
    request: SimulationRunCreate,

    db: Session = Depends(get_db)
):
    """
    Start a new simulation run.
    """
    global _running_simulation

    if _running_simulation:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": f"Simulation already running: {_running_simulation}"
            }
        )

    run_id = str(uuid.uuid4())
    config = {
        "numPlayers": request.num_players,
        "targetGames": request.target_games,
        "outputDir": request.output_dir,
        "mctsSimulations": request.mcts_simulations,
        "useNeural": request.use_neural
    }

    run = SimulationRunDB(
        id=run_id,
        start_time=datetime.utcnow(),
        status="running",
        num_players=request.num_players,
        target_games=request.target_games,
        config=json.dumps(config),
        data_path=str(DATA_DIR / request.output_dir)
    )

    db.add(run)
    db.commit()

    _running_simulation = run_id

    # Start background simulation
    from ..workers.simulation_worker import run_simulation
    asyncio.create_task(run_simulation(run_id, config))

    return {
        "success": True,
        "data": {
            "runId": run_id,
            "status": "running",
            "streamUrl": f"/api/simulation/stream/{run_id}"
        }
    }


@router.post("/stop/{run_id}")
async def stop_simulation(run_id: str, db: Session = Depends(get_db)):
    """
    Stop a running simulation.
    """
    global _running_simulation

    run = db.query(SimulationRunDB).filter(SimulationRunDB.id == run_id).first()

    if not run:
        raise HTTPException(status_code=404, detail="Simulation run not found")

    if run.status != "running":
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": f"Simulation is not running (status: {run.status})"
            }
        )

    # Signal cancellation
    from ..workers.simulation_worker import cancel_simulation
    cancel_simulation(run_id)

    run.status = "cancelled"
    run.end_time = datetime.utcnow()
    db.commit()

    if _running_simulation == run_id:
        _running_simulation = None

    return {
        "success": True,
        "data": {
            "runId": run_id,
            "status": "cancelled",
            "gamesCompleted": run.games_completed,
            "positionsGenerated": run.positions_generated
        }
    }


@router.get("/stream/{run_id}")
async def simulation_stream(run_id: str, request: Request):
    """
    SSE stream for live simulation progress.
    """
    return EventSourceResponse(
        stream_generator(run_id, sse_manager, request)
    )


@router.delete("/runs/{run_id}")
async def delete_simulation_run(run_id: str, db: Session = Depends(get_db)):
    """
    Delete a simulation run. Cannot delete running simulations.
    """
    global _running_simulation

    run = db.query(SimulationRunDB).filter(SimulationRunDB.id == run_id).first()

    if not run:
        raise HTTPException(status_code=404, detail="Simulation run not found")

    if run.status == "running":
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "Cannot delete a running simulation. Stop it first."
            }
        )

    # Optionally delete data directory
    import shutil
    from pathlib import Path
    if run.data_path:
        data_path = Path(run.data_path)
        if data_path.exists() and data_path.is_dir():
            try:
                shutil.rmtree(data_path)
            except Exception as e:
                # Log but dont fail if we cant delete files
                print(f"Warning: Could not delete data directory: {e}")

    # Delete from database
    db.delete(run)
    db.commit()

    return {
        "success": True,
        "data": {
            "deleted": run_id,
            "dataPathRemoved": run.data_path
        }
    }
