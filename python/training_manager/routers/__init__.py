"""API routers for Training Manager."""

from .system import router as system_router
from .dashboard import router as dashboard_router
from .simulation import router as simulation_router
from .training import router as training_router
from .benchmark import router as benchmark_router
from .models_router import router as models_router

__all__ = [
    "system_router",
    "dashboard_router",
    "simulation_router",
    "training_router",
    "benchmark_router",
    "models_router"
]
