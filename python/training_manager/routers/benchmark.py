"""
Benchmark endpoints for model comparison.
"""

import uuid
import json
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from ..database.connection import get_db
from ..database.models import BenchmarkRunDB, ModelDB, BenchmarkRunCreate
from ..services.sse_manager import sse_manager, stream_generator

router = APIRouter(prefix="/api/benchmark", tags=["benchmark"])

# Track running benchmark
_running_benchmark: Optional[str] = None


@router.get("/runs")
async def list_benchmark_runs(
    limit: int = 20,
    offset: int = 0,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    List all benchmark runs with pagination.
    """
    query = db.query(BenchmarkRunDB)

    if status:
        query = query.filter(BenchmarkRunDB.status == status)

    total = query.count()
    runs = query.order_by(
        BenchmarkRunDB.created_at.desc()
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
                    "modelAId": r.model_a_id,
                    "modelBId": r.model_b_id,
                    "modelAName": r.model_a_name,
                    "modelBName": r.model_b_name,
                    "numPlayers": r.num_players,
                    "targetGames": r.target_games,
                    "gamesCompleted": r.games_completed,
                    "modelAWins": r.model_a_wins,
                    "modelBWins": r.model_b_wins,
                    "draws": r.draws,
                    "winner": r.winner,
                    "winRateA": r.win_rate_a,
                    "winRateB": r.win_rate_b,
                    "createdAt": r.created_at.isoformat()
                }
                for r in runs
            ],
            "total": total,
            "hasMore": offset + limit < total
        }
    }


@router.get("/runs/{run_id}")
async def get_benchmark_run(run_id: str, db: Session = Depends(get_db)):
    """
    Get details of a specific benchmark run.
    """
    run = db.query(BenchmarkRunDB).filter(BenchmarkRunDB.id == run_id).first()

    if not run:
        raise HTTPException(status_code=404, detail="Benchmark run not found")

    game_details = None
    if run.game_details:
        try:
            game_details = json.loads(run.game_details)
        except:
            pass

    return {
        "success": True,
        "data": {
            "id": run.id,
            "startTime": run.start_time.isoformat() if run.start_time else None,
            "endTime": run.end_time.isoformat() if run.end_time else None,
            "status": run.status,
            "modelA": {"id": run.model_a_id, "name": run.model_a_name},
            "modelB": {"id": run.model_b_id, "name": run.model_b_name},
            "numPlayers": run.num_players,
            "targetGames": run.target_games,
            "gamesCompleted": run.games_completed,
            "results": {
                "modelAWins": run.model_a_wins,
                "modelBWins": run.model_b_wins,
                "draws": run.draws,
                "modelAWinRate": run.win_rate_a,
                "modelBWinRate": run.win_rate_b,
                "winner": run.winner
            },
            "gameDetails": game_details,
            "errorMessage": run.error_message,
            "createdAt": run.created_at.isoformat()
        }
    }


@router.post("/start")
async def start_benchmark(
    request: BenchmarkRunCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Start a new benchmark run.
    """
    global _running_benchmark

    if _running_benchmark:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": f"Benchmark already running: {_running_benchmark}"
            }
        )

    # Validate models exist
    model_a = db.query(ModelDB).filter(ModelDB.id == request.model_a_id).first()
    model_b = db.query(ModelDB).filter(ModelDB.id == request.model_b_id).first()

    if not model_a:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": f"Model A not found: {request.model_a_id}"}
        )

    if not model_b:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": f"Model B not found: {request.model_b_id}"}
        )

    run_id = str(uuid.uuid4())
    config = {
        "modelAId": request.model_a_id,
        "modelBId": request.model_b_id,
        "numPlayers": request.num_players,
        "targetGames": request.target_games
    }

    run = BenchmarkRunDB(
        id=run_id,
        start_time=datetime.utcnow(),
        status="running",
        model_a_id=request.model_a_id,
        model_b_id=request.model_b_id,
        model_a_name=model_a.name,
        model_b_name=model_b.name,
        num_players=request.num_players,
        target_games=request.target_games,
        config=json.dumps(config)
    )

    db.add(run)
    db.commit()

    _running_benchmark = run_id

    # Start background benchmark
    from ..workers.benchmark_worker import run_benchmark
    background_tasks.add_task(
        run_benchmark,
        run_id,
        model_a.file_path,
        model_b.file_path,
        config
    )

    return {
        "success": True,
        "data": {
            "runId": run_id,
            "status": "running",
            "streamUrl": f"/api/benchmark/stream/{run_id}"
        }
    }


@router.post("/stop/{run_id}")
async def stop_benchmark(run_id: str, db: Session = Depends(get_db)):
    """
    Stop a running benchmark.
    """
    global _running_benchmark

    run = db.query(BenchmarkRunDB).filter(BenchmarkRunDB.id == run_id).first()

    if not run:
        raise HTTPException(status_code=404, detail="Benchmark run not found")

    if run.status != "running":
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": f"Benchmark is not running (status: {run.status})"
            }
        )

    # Signal cancellation
    from ..workers.benchmark_worker import cancel_benchmark
    cancel_benchmark(run_id)

    run.status = "cancelled"
    run.end_time = datetime.utcnow()
    db.commit()

    if _running_benchmark == run_id:
        _running_benchmark = None

    return {
        "success": True,
        "data": {
            "runId": run_id,
            "status": "cancelled",
            "gamesCompleted": run.games_completed
        }
    }


@router.delete("/runs/{run_id}")
async def delete_benchmark(run_id: str, db: Session = Depends(get_db)):
    """
    Delete a benchmark result.
    """
    run = db.query(BenchmarkRunDB).filter(BenchmarkRunDB.id == run_id).first()

    if not run:
        raise HTTPException(status_code=404, detail="Benchmark run not found")

    if run.status == "running":
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "Cannot delete running benchmark. Stop it first."
            }
        )

    db.delete(run)
    db.commit()

    return {
        "success": True,
        "data": {
            "deleted": True,
            "runId": run_id
        }
    }


@router.delete("/reset")
async def reset_all_benchmarks(db: Session = Depends(get_db)):
    """
    Delete all benchmark runs (except currently running ones).
    """
    global _running_benchmark

    # Check if any benchmark is running
    running = db.query(BenchmarkRunDB).filter(BenchmarkRunDB.status == "running").first()

    if running:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "Cannot reset while benchmark is running. Stop it first."
            }
        )

    # Delete all benchmark runs
    count = db.query(BenchmarkRunDB).delete()
    db.commit()

    _running_benchmark = None

    return {
        "success": True,
        "data": {
            "deleted": count,
            "message": f"Deleted {count} benchmark run(s)"
        }
    }


@router.get("/stream/{run_id}")
async def benchmark_stream(run_id: str, request: Request):
    """
    SSE stream for live benchmark progress.
    """
    return EventSourceResponse(
        stream_generator(run_id, sse_manager, request)
    )
