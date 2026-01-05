"""
Dashboard endpoints for stats and overview.
"""

from datetime import datetime, timedelta
from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from ..database.connection import get_db
from ..database.models import (
    SimulationRunDB, TrainingRunDB, BenchmarkRunDB, ModelDB, DashboardStatsDB
)

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/stats")
async def get_dashboard_stats(db: Session = Depends(get_db)):
    """
    Get dashboard overview statistics.
    """
    # Get active model
    active_model = db.query(ModelDB).filter(ModelDB.is_active == True).first()

    # Get simulation stats
    total_games = db.query(func.sum(SimulationRunDB.games_completed)).scalar() or 0
    total_positions = db.query(func.sum(SimulationRunDB.positions_generated)).scalar() or 0

    # Today's stats
    today = datetime.utcnow().date()
    today_start = datetime.combine(today, datetime.min.time())

    games_today = db.query(func.sum(SimulationRunDB.games_completed)).filter(
        SimulationRunDB.created_at >= today_start
    ).scalar() or 0

    positions_today = db.query(func.sum(SimulationRunDB.positions_generated)).filter(
        SimulationRunDB.created_at >= today_start
    ).scalar() or 0

    # Training stats
    total_training = db.query(func.count(TrainingRunDB.id)).scalar() or 0
    last_training = db.query(TrainingRunDB).order_by(
        TrainingRunDB.created_at.desc()
    ).first()

    best_loss = db.query(func.min(TrainingRunDB.best_val_loss)).filter(
        TrainingRunDB.status == "completed"
    ).scalar()

    # Benchmark stats
    total_benchmarks = db.query(func.count(BenchmarkRunDB.id)).scalar() or 0
    last_benchmark = db.query(BenchmarkRunDB).order_by(
        BenchmarkRunDB.created_at.desc()
    ).first()

    return {
        "success": True,
        "data": {
            "activeModel": {
                "name": active_model.name if active_model else None,
                "accuracy": active_model.accuracy if active_model else None,
                "valLoss": active_model.val_loss if active_model else None,
                "createdAt": active_model.created_at.isoformat() if active_model else None
            } if active_model else None,
            "simulations": {
                "totalGames": total_games,
                "totalPositions": total_positions,
                "gamesToday": games_today,
                "positionsToday": positions_today
            },
            "training": {
                "totalRuns": total_training,
                "lastRunAt": last_training.created_at.isoformat() if last_training else None,
                "bestLoss": best_loss
            },
            "benchmarks": {
                "totalRuns": total_benchmarks,
                "lastRunAt": last_benchmark.created_at.isoformat() if last_benchmark else None
            }
        }
    }


@router.get("/recent-activity")
async def get_recent_activity(
    limit: int = 5,
    db: Session = Depends(get_db)
):
    """
    Get recent activity across all sections.
    """
    # Recent simulations
    recent_sims = db.query(SimulationRunDB).order_by(
        SimulationRunDB.created_at.desc()
    ).limit(limit).all()

    # Recent training runs
    recent_training = db.query(TrainingRunDB).order_by(
        TrainingRunDB.created_at.desc()
    ).limit(limit).all()

    # Recent benchmarks
    recent_benchmarks = db.query(BenchmarkRunDB).order_by(
        BenchmarkRunDB.created_at.desc()
    ).limit(limit).all()

    return {
        "success": True,
        "data": {
            "recentSimulations": [
                {
                    "id": s.id,
                    "status": s.status,
                    "gamesCompleted": s.games_completed,
                    "targetGames": s.target_games,
                    "createdAt": s.created_at.isoformat()
                }
                for s in recent_sims
            ],
            "recentTrainingRuns": [
                {
                    "id": t.id,
                    "status": t.status,
                    "epochsCompleted": t.epochs_completed,
                    "epochsTotal": t.epochs_total,
                    "bestValLoss": t.best_val_loss,
                    "createdAt": t.created_at.isoformat()
                }
                for t in recent_training
            ],
            "recentBenchmarks": [
                {
                    "id": b.id,
                    "status": b.status,
                    "modelAName": b.model_a_name,
                    "modelBName": b.model_b_name,
                    "gamesCompleted": b.games_completed,
                    "winner": b.winner,
                    "createdAt": b.created_at.isoformat()
                }
                for b in recent_benchmarks
            ]
        }
    }
