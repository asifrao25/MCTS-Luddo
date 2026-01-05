"""
Luddo AI Training Manager Service.

FastAPI application providing REST API + SSE for:
- Simulation: Generate training data with live progress
- Training: Manual MLX training with epoch-by-epoch progress
- Benchmarking: Model comparison with history
- Dashboard: Stats and active model info
- System: Health, metrics, real-time monitoring
"""

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import PORT, APP_NAME, APP_VERSION
from .database.connection import init_db
from .routers import (
    system_router,
    dashboard_router,
    simulation_router,
    training_router,
    benchmark_router,
    models_router
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    print(f"Starting {APP_NAME} v{APP_VERSION} on port {PORT}")
    init_db()
    print("Database initialized")

    yield

    # Shutdown
    print("Shutting down...")


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="Training management API for Luddo AI",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(dashboard_router)
app.include_router(simulation_router)
app.include_router(training_router)
app.include_router(benchmark_router)
app.include_router(models_router)
app.include_router(system_router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": APP_NAME,
        "version": APP_VERSION,
        "status": "running"
    }


def main():
    """Entry point for running the application."""
    uvicorn.run(
        "training_manager.main:app",
        host="0.0.0.0",
        port=PORT,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
