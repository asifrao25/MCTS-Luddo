"""
SQLAlchemy models and Pydantic schemas for the Training Manager.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from pydantic import BaseModel, Field

from .connection import Base


# ============== Enums ==============

class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ============== SQLAlchemy Models ==============

class SimulationRunDB(Base):
    __tablename__ = "simulation_runs"

    id = Column(String, primary_key=True)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=True)
    status = Column(String, default="pending")
    num_players = Column(Integer, default=4)
    target_games = Column(Integer, nullable=False)
    games_completed = Column(Integer, default=0)
    positions_generated = Column(Integer, default=0)
    config = Column(Text, nullable=False)  # JSON
    data_path = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class TrainingRunDB(Base):
    __tablename__ = "training_runs"

    id = Column(String, primary_key=True)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=True)
    status = Column(String, default="pending")
    epochs_total = Column(Integer, nullable=False)
    epochs_completed = Column(Integer, default=0)
    current_epoch = Column(Integer, default=0)
    batch_size = Column(Integer, nullable=False)
    learning_rate = Column(Float, nullable=False)
    validation_split = Column(Float, default=0.1)
    train_loss = Column(Float, nullable=True)
    val_loss = Column(Float, nullable=True)
    best_val_loss = Column(Float, nullable=True)
    best_epoch = Column(Integer, nullable=True)
    train_samples = Column(Integer, nullable=True)
    val_samples = Column(Integer, nullable=True)
    data_sources = Column(Text, nullable=False)  # JSON array
    model_path = Column(String, nullable=True)
    config = Column(Text, nullable=False)  # JSON
    loss_history = Column(Text, nullable=True)  # JSON array
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelDB(Base):
    __tablename__ = "models"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    version = Column(Integer, default=1)
    training_run_id = Column(String, ForeignKey("training_runs.id"), nullable=True)
    file_path = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=False)
    accuracy = Column(Float, nullable=True)
    val_loss = Column(Float, nullable=True)
    train_samples = Column(Integer, nullable=True)
    config = Column(Text, nullable=True)  # JSON


class BenchmarkRunDB(Base):
    __tablename__ = "benchmark_runs"

    id = Column(String, primary_key=True)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=True)
    status = Column(String, default="pending")
    model_a_id = Column(String, nullable=False)
    model_b_id = Column(String, nullable=False)
    model_a_name = Column(String, nullable=False)
    model_b_name = Column(String, nullable=False)
    num_players = Column(Integer, default=4)
    target_games = Column(Integer, nullable=False)
    games_completed = Column(Integer, default=0)
    model_a_wins = Column(Integer, default=0)
    model_b_wins = Column(Integer, default=0)
    draws = Column(Integer, default=0)
    winner = Column(String, nullable=True)
    win_rate_a = Column(Float, nullable=True)
    win_rate_b = Column(Float, nullable=True)
    config = Column(Text, nullable=False)  # JSON
    game_details = Column(Text, nullable=True)  # JSON array
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class DashboardStatsDB(Base):
    __tablename__ = "dashboard_stats"

    id = Column(Integer, primary_key=True)
    total_games_simulated = Column(Integer, default=0)
    total_positions_generated = Column(Integer, default=0)
    games_today = Column(Integer, default=0)
    positions_today = Column(Integer, default=0)
    total_training_runs = Column(Integer, default=0)
    total_benchmarks = Column(Integer, default=0)
    active_model_name = Column(String, nullable=True)
    active_model_accuracy = Column(Float, nullable=True)
    last_updated = Column(DateTime, default=datetime.utcnow)


# ============== Pydantic Schemas ==============

# --- Simulation ---

class SimulationRunCreate(BaseModel):
    num_players: int = Field(default=4, ge=2, le=4)
    target_games: int = Field(default=10000, ge=1)
    output_dir: str = Field(default="self_play_games")
    mcts_simulations: int = Field(default=20, ge=1, le=500)
    use_neural: bool = Field(default=False)


class SimulationRun(BaseModel):
    id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str
    num_players: int
    target_games: int
    games_completed: int
    positions_generated: int
    data_path: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


# --- Training ---

class TrainingRunCreate(BaseModel):
    epochs: int = Field(default=100, ge=1)
    batch_size: int = Field(default=256, ge=1)
    learning_rate: float = Field(default=0.001, gt=0)
    validation_split: float = Field(default=0.1, ge=0, le=0.5)
    data_sources: List[str] = Field(default=["4player_games"])
    model_name: Optional[str] = None


class TrainingRun(BaseModel):
    id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str
    epochs_total: int
    epochs_completed: int
    current_epoch: int
    batch_size: int
    learning_rate: float
    validation_split: float
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    best_val_loss: Optional[float] = None
    best_epoch: Optional[int] = None
    train_samples: Optional[int] = None
    val_samples: Optional[int] = None
    model_path: Optional[str] = None
    loss_history: Optional[List[Dict[str, Any]]] = None
    error_message: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


# --- Model ---

class ModelCreate(BaseModel):
    name: str
    file_path: str
    training_run_id: Optional[str] = None
    accuracy: Optional[float] = None
    val_loss: Optional[float] = None


class Model(BaseModel):
    id: str
    name: str
    version: int
    training_run_id: Optional[str] = None
    file_path: str
    created_at: datetime
    is_active: bool
    accuracy: Optional[float] = None
    val_loss: Optional[float] = None
    train_samples: Optional[int] = None

    class Config:
        from_attributes = True


# --- Benchmark ---

class BenchmarkRunCreate(BaseModel):
    model_a_id: str
    model_b_id: str
    num_players: int = Field(default=4, ge=2, le=4)
    target_games: int = Field(default=100, ge=1)


class BenchmarkRun(BaseModel):
    id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str
    model_a_id: str
    model_b_id: str
    model_a_name: str
    model_b_name: str
    num_players: int
    target_games: int
    games_completed: int
    model_a_wins: int
    model_b_wins: int
    draws: int
    winner: Optional[str] = None
    win_rate_a: Optional[float] = None
    win_rate_b: Optional[float] = None
    game_details: Optional[List[Dict[str, Any]]] = None
    error_message: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


# --- Dashboard ---

class DashboardStats(BaseModel):
    total_games_simulated: int
    total_positions_generated: int
    games_today: int
    positions_today: int
    total_training_runs: int
    total_benchmarks: int
    active_model_name: Optional[str] = None
    active_model_accuracy: Optional[float] = None
    last_updated: datetime

    class Config:
        from_attributes = True


# --- API Responses ---

class APIResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None


class PaginatedResponse(BaseModel):
    success: bool
    data: Dict[str, Any]
    total: int
    has_more: bool
