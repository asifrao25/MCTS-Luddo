"""
Training endpoints for MLX model training.
"""

import os
import uuid
import json
import asyncio
from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from ..database.connection import get_db
from ..database.models import TrainingRunDB, TrainingRunCreate
from ..services.sse_manager import sse_manager, stream_generator
from ..config import DATA_DIR, MODELS_DIR

router = APIRouter(prefix="/api/training", tags=["training"])

# Track running training with lock to prevent race conditions
_running_training: Optional[str] = None
_training_lock = asyncio.Lock()


@router.get("/runs")
async def list_training_runs(
    limit: int = 20,
    offset: int = 0,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    List all training runs with pagination.
    """
    query = db.query(TrainingRunDB)

    if status:
        query = query.filter(TrainingRunDB.status == status)

    total = query.count()
    runs = query.order_by(
        TrainingRunDB.created_at.desc()
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
                    "epochsTotal": r.epochs_total,
                    "epochsCompleted": r.epochs_completed,
                    "batchSize": r.batch_size,
                    "learningRate": r.learning_rate,
                    "bestValLoss": r.best_val_loss,
                    "bestEpoch": r.best_epoch,
                    "modelPath": r.model_path,
                    "createdAt": r.created_at.isoformat()
                }
                for r in runs
            ],
            "total": total,
            "hasMore": offset + limit < total
        }
    }


@router.get("/runs/{run_id}")
async def get_training_run(run_id: str, db: Session = Depends(get_db)):
    """
    Get details of a specific training run including loss history.
    """
    run = db.query(TrainingRunDB).filter(TrainingRunDB.id == run_id).first()

    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")

    loss_history = None
    if run.loss_history:
        try:
            loss_history = json.loads(run.loss_history)
        except:
            pass

    data_sources = []
    if run.data_sources:
        try:
            data_sources = json.loads(run.data_sources)
        except:
            pass

    return {
        "success": True,
        "data": {
            "id": run.id,
            "startTime": run.start_time.isoformat() if run.start_time else None,
            "endTime": run.end_time.isoformat() if run.end_time else None,
            "status": run.status,
            "epochsTotal": run.epochs_total,
            "epochsCompleted": run.epochs_completed,
            "currentEpoch": run.current_epoch,
            "batchSize": run.batch_size,
            "learningRate": run.learning_rate,
            "validationSplit": run.validation_split,
            "trainLoss": run.train_loss,
            "valLoss": run.val_loss,
            "bestValLoss": run.best_val_loss,
            "bestEpoch": run.best_epoch,
            "trainSamples": run.train_samples,
            "valSamples": run.val_samples,
            "dataSources": data_sources,
            "modelPath": run.model_path,
            "lossHistory": loss_history,
            "errorMessage": run.error_message,
            "createdAt": run.created_at.isoformat()
        }
    }


@router.post("/start")
async def start_training(
    request: TrainingRunCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Start a new training run (manual trigger).
    """
    global _running_training

    # Check if training already running (with lock)
    async with _training_lock:
        if _running_training:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": f"Training already running: {_running_training}"
                }
            )
        # Reserve the slot immediately to prevent race conditions
        run_id = str(uuid.uuid4())
        _running_training = run_id

    # Validate data sources exist (outside lock for performance)
    for source in request.data_sources:
        source_path = DATA_DIR / source
        if not source_path.exists():
            # Release the slot on validation failure
            async with _training_lock:
                _running_training = None
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": f"Data source not found: {source}"
                }
            )

    config = {
        "epochs": request.epochs,
        "batchSize": request.batch_size,
        "learningRate": request.learning_rate,
        "validationSplit": request.validation_split,
        "dataSources": request.data_sources,
        "modelName": request.model_name or f"model_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    }

    run = TrainingRunDB(
        id=run_id,
        start_time=datetime.utcnow(),
        status="running",
        epochs_total=request.epochs,
        batch_size=request.batch_size,
        learning_rate=request.learning_rate,
        validation_split=request.validation_split,
        data_sources=json.dumps(request.data_sources),
        config=json.dumps(config)
    )

    db.add(run)
    db.commit()

    # Start background training
    from ..workers.training_worker import run_training
    background_tasks.add_task(run_training, run_id, config)

    return {
        "success": True,
        "data": {
            "runId": run_id,
            "status": "running",
            "streamUrl": f"/api/training/stream/{run_id}"
        }
    }


@router.post("/stop/{run_id}")
async def stop_training(run_id: str, db: Session = Depends(get_db)):
    """
    Stop a running training (saves current best).
    """
    global _running_training

    run = db.query(TrainingRunDB).filter(TrainingRunDB.id == run_id).first()

    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")

    if run.status != "running":
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": f"Training is not running (status: {run.status})"
            }
        )

    # Signal cancellation
    from ..workers.training_worker import cancel_training
    cancel_training(run_id)

    run.status = "cancelled"
    run.end_time = datetime.utcnow()
    db.commit()

    # Clear running training with lock
    async with _training_lock:
        if _running_training == run_id:
            _running_training = None

    return {
        "success": True,
        "data": {
            "runId": run_id,
            "status": "cancelled",
            "epochsCompleted": run.epochs_completed,
            "bestValLoss": run.best_val_loss
        }
    }


@router.get("/stream/{run_id}")
async def training_stream(run_id: str, request: Request):
    """
    SSE stream for live training progress.
    """
    return EventSourceResponse(
        stream_generator(run_id, sse_manager, request)
    )


@router.get("/data-sources")
async def list_data_sources():
    """
    List available training data sources.
    """
    sources = []

    for item in DATA_DIR.iterdir():
        if item.is_dir():
            # Look for metadata files
            metadata_files = list(item.glob("metadata_*.json"))
            if metadata_files:
                with open(metadata_files[-1]) as f:
                    metadata = json.load(f)

                sources.append({
                    "name": item.name,
                    "path": str(item),
                    "numGames": metadata.get("num_games", 0),
                    "numPositions": metadata.get("num_positions", 0),
                    "numPlayers": metadata.get("num_players", 2),
                    "timestamp": metadata.get("timestamp", "unknown")
                })

    return {
        "success": True,
        "data": {
            "sources": sources
        }
    }
