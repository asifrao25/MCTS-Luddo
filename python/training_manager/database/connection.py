"""
SQLite database connection and initialization.
"""

import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import StaticPool

from ..config import DATABASE_PATH

# Ensure data directory exists
DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

# SQLite connection string
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

# Create engine with SQLite-specific settings
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
    echo=False
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def get_db():
    """Dependency for getting database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database with all tables."""
    from . import models  # Import to register models

    # Create all tables
    Base.metadata.create_all(bind=engine)

    # Initialize dashboard stats if not exists
    with SessionLocal() as db:
        result = db.execute(text("SELECT COUNT(*) FROM dashboard_stats")).scalar()
        if result == 0:
            db.execute(text("""
                INSERT INTO dashboard_stats (
                    id, total_games_simulated, total_positions_generated,
                    games_today, positions_today, total_training_runs,
                    total_benchmarks, last_updated
                ) VALUES (1, 0, 0, 0, 0, 0, 0, CURRENT_TIMESTAMP)
            """))
            db.commit()

    print(f"[Database] Initialized at {DATABASE_PATH}")
