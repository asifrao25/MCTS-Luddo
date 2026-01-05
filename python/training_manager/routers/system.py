"""
System endpoints: health check, neural reload, system metrics.
"""

import asyncio
from datetime import datetime
from typing import Optional

import httpx
from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from sqlalchemy.orm import Session

from ..config import AI_ENGINE_URL, NEURAL_SERVER_URL
from ..services.metrics_service import get_system_metrics
from ..database.connection import get_db
from ..database.models import SimulationRunDB, TrainingRunDB, BenchmarkRunDB

router = APIRouter(prefix="/api/system", tags=["system"])


def get_active_tasks(db: Session):
    """Get currently running tasks from database."""
    active = {
        "simulation": None,
        "training": None,
        "benchmark": None
    }

    # Check for running simulation
    sim = db.query(SimulationRunDB).filter(SimulationRunDB.status == "running").first()
    if sim:
        active["simulation"] = sim.id

    # Check for running training
    train = db.query(TrainingRunDB).filter(TrainingRunDB.status == "running").first()
    if train:
        active["training"] = train.id

    # Check for running benchmark
    bench = db.query(BenchmarkRunDB).filter(BenchmarkRunDB.status == "running").first()
    if bench:
        active["benchmark"] = bench.id

    return active


@router.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """
    Health check endpoint with service status and active tasks.
    """
    # Check AI Engine
    ai_engine_status = "unknown"
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{AI_ENGINE_URL}/api/ai/health")
            if response.status_code == 200:
                ai_engine_status = "healthy"
            else:
                ai_engine_status = "unhealthy"
    except Exception:
        ai_engine_status = "unreachable"

    # Check Neural Server
    neural_status = "unknown"
    neural_model_loaded = False
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{NEURAL_SERVER_URL}/health")
            if response.status_code == 200:
                data = response.json()
                neural_status = "healthy" if data.get("status") == "healthy" else "unhealthy"
                neural_model_loaded = data.get("model_loaded", False)
            else:
                neural_status = "unhealthy"
    except Exception:
        neural_status = "unreachable"

    # Get active tasks from database
    active_tasks = get_active_tasks(db)

    return {
        "success": True,
        "data": {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "aiEngine": {"status": ai_engine_status, "port": 3020},
                "neuralServer": {
                    "status": neural_status,
                    "port": 3021,
                    "modelLoaded": neural_model_loaded
                },
                "trainingManager": {"status": "healthy", "port": 3022}
            },
            "activeTasks": active_tasks
        }
    }


@router.post("/reload-neural")
async def reload_neural_server():
    """
    Force reload the neural server with current active model.
    """
    import subprocess

    try:
        result = subprocess.run(
            ["pm2", "restart", "luddo-neural-eval"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            return {
                "success": True,
                "data": {
                    "reloaded": True,
                    "message": "Neural server restart initiated"
                }
            }
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": f"Failed to restart: {result.stderr}"
                }
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )


@router.get("/metrics")
async def get_metrics():
    """
    Get current system metrics snapshot.
    """
    metrics = await get_system_metrics()
    return {
        "success": True,
        "data": metrics
    }


@router.get("/metrics/stream")
async def metrics_stream(request: Request):
    """
    SSE stream for live system metrics (updates every 1 second).
    """
    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break

                metrics = await get_system_metrics()
                yield {
                    "event": "metrics",
                    "data": str(metrics).replace("'", '"')
                }
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass

    return EventSourceResponse(event_generator())
