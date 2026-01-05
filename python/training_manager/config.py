"""
Configuration settings for the Training Manager service.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
PYTHON_DIR = BASE_DIR / "python"
TRAINING_MANAGER_DIR = PYTHON_DIR / "training_manager"

# Data paths
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(BASE_DIR / "models")))
DATABASE_PATH = Path(os.getenv(
    "DATABASE_PATH",
    str(TRAINING_MANAGER_DIR / "data" / "training_manager.db")
))

# Application info
APP_NAME = "Luddo Training Manager"
APP_VERSION = "1.0.0"

# Server configuration
PORT = int(os.getenv("PORT", 3022))
HOST = os.getenv("HOST", "0.0.0.0")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# External services
AI_ENGINE_URL = os.getenv("AI_ENGINE_URL", "http://localhost:3020")
NEURAL_SERVER_URL = os.getenv("NEURAL_SERVER_URL", "http://localhost:3021")

# Simulation defaults
DEFAULT_NUM_PLAYERS = 4
DEFAULT_TARGET_GAMES = 10000
MAX_MOVES_PER_GAME = 500

# Training defaults
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 256
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_VALIDATION_SPLIT = 0.1

# Benchmark defaults
DEFAULT_BENCHMARK_GAMES = 100

# SSE settings
SSE_KEEPALIVE_INTERVAL = 30  # seconds
METRICS_STREAM_INTERVAL = 1  # seconds

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
