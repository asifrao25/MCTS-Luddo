"""
Models registry endpoints.
"""

import os
import re
import uuid
import json
import subprocess
from datetime import datetime
from typing import Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from ..database.connection import get_db
from ..database.models import ModelDB, ModelCreate, Model
from ..config import MODELS_DIR, NEURAL_SERVER_URL

router = APIRouter(prefix="/api/models", tags=["models"])


def parse_version(name: str) -> Tuple[int, int]:
    """Parse version from model name like 'mcts_v1.2' -> (1, 2)"""
    match = re.search(r'_v(\d+)\.(\d+)$', name)
    if match:
        return int(match.group(1)), int(match.group(2))
    # Try integer version format like 'mcts_v1'
    match = re.search(r'_v(\d+)$', name)
    if match:
        return int(match.group(1)), 0
    return 1, 0


def get_next_version(db: Session, base_name: str) -> str:
    """Get next version string for a model base name."""
    # Find all models with this base name
    existing = db.query(ModelDB).filter(
        ModelDB.name.like(f"{base_name}_v%")
    ).all()

    if not existing:
        return "1.0"

    # Parse all versions and find the highest
    max_major, max_minor = 0, 0
    for model in existing:
        major, minor = parse_version(model.name)
        if major > max_major or (major == max_major and minor > max_minor):
            max_major, max_minor = major, minor

    # Increment minor version
    return f"{max_major}.{max_minor + 1}"


@router.get("")
async def list_models(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """
    List all registered models.
    """
    models = db.query(ModelDB).order_by(
        ModelDB.created_at.desc()
    ).offset(offset).limit(limit).all()

    total = db.query(ModelDB).count()

    return {
        "success": True,
        "data": {
            "models": [
                {
                    "id": m.id,
                    "name": m.name,
                    "version": m.version,
                    "isActive": m.is_active,
                    "accuracy": m.accuracy,
                    "valLoss": m.val_loss,
                    "trainSamples": m.train_samples,
                    "trainingRunId": m.training_run_id,
                    "filePath": m.file_path,
                    "createdAt": m.created_at.isoformat()
                }
                for m in models
            ],
            "total": total,
            "hasMore": offset + limit < total
        }
    }


@router.get("/{model_id}")
async def get_model(model_id: str, db: Session = Depends(get_db)):
    """
    Get model details.
    """
    model = db.query(ModelDB).filter(ModelDB.id == model_id).first()

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    config = None
    if model.config:
        try:
            config = json.loads(model.config)
        except:
            pass

    return {
        "success": True,
        "data": {
            "id": model.id,
            "name": model.name,
            "version": model.version,
            "isActive": model.is_active,
            "accuracy": model.accuracy,
            "valLoss": model.val_loss,
            "trainSamples": model.train_samples,
            "trainingRunId": model.training_run_id,
            "filePath": model.file_path,
            "config": config,
            "createdAt": model.created_at.isoformat()
        }
    }


@router.post("/{model_id}/activate")
async def activate_model(model_id: str, db: Session = Depends(get_db)):
    """
    Set a model as the active model and reload the neural server.
    """
    model = db.query(ModelDB).filter(ModelDB.id == model_id).first()

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Check if model file exists
    if not os.path.exists(model.file_path):
        raise HTTPException(status_code=400, detail=f"Model file not found: {model.file_path}")

    # Deactivate all models
    db.query(ModelDB).update({ModelDB.is_active: False})

    # Activate this model
    model.is_active = True
    db.commit()

    # Copy model to best_model.npz location
    best_model_path = os.path.join(MODELS_DIR, "best_model.npz")
    try:
        import shutil
        shutil.copy2(model.file_path, best_model_path)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"Failed to copy model: {str(e)}"
            }
        )

    # Restart neural server
    neural_reloaded = False
    try:
        result = subprocess.run(
            ["pm2", "restart", "luddo-neural-eval"],
            capture_output=True,
            text=True,
            timeout=10
        )
        neural_reloaded = result.returncode == 0
    except Exception:
        pass

    return {
        "success": True,
        "data": {
            "modelId": model.id,
            "name": model.name,
            "activated": True,
            "neuralServerReloaded": neural_reloaded
        }
    }


@router.delete("/{model_id}")
async def delete_model(model_id: str, db: Session = Depends(get_db)):
    """
    Delete a model (cannot delete active model).
    """
    model = db.query(ModelDB).filter(ModelDB.id == model_id).first()

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    if model.is_active:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete active model. Activate another model first."
        )

    # Delete model file if it exists
    if os.path.exists(model.file_path):
        try:
            os.remove(model.file_path)
        except Exception:
            pass  # Continue even if file deletion fails

    db.delete(model)
    db.commit()

    return {
        "success": True,
        "data": {
            "deleted": True,
            "modelId": model_id
        }
    }


def register_model(
    db: Session,
    name: str,
    file_path: str,
    training_run_id: Optional[str] = None,
    accuracy: Optional[float] = None,
    val_loss: Optional[float] = None,
    train_samples: Optional[int] = None,
    config: Optional[dict] = None
) -> ModelDB:
    """
    Register a new model in the database with semantic versioning.
    Used by training service after training completes.

    Versioning: mcts_v1.0, mcts_v1.1, mcts_v1.2, etc.
    Each training run increments the minor version.
    """
    # Extract base name (default to "mcts" if name has no pattern)
    base_name = name.split("_v")[0] if "_v" in name else name

    # Get next semantic version
    version_str = get_next_version(db, base_name)
    major, minor = map(int, version_str.split("."))

    # Create model with semantic version name
    model = ModelDB(
        id=str(uuid.uuid4()),
        name=f"{base_name}_v{version_str}",  # e.g., "mcts_v1.2"
        version=major * 100 + minor,  # Store as integer for sorting (102 = v1.2)
        training_run_id=training_run_id,
        file_path=file_path,
        accuracy=accuracy,
        val_loss=val_loss,
        train_samples=train_samples,
        config=json.dumps(config) if config else None,
        is_active=False,
        created_at=datetime.utcnow()
    )

    db.add(model)
    db.commit()
    db.refresh(model)

    return model


@router.post("/register")
async def register_model_endpoint(
    name: str = "mcts",
    file_path: str = "/Users/m4-mac/mac-luddo/models/best_model.npz",
    val_loss: float = None,
    accuracy: float = None,
    train_samples: int = None,
    db: Session = Depends(get_db)
):
    """
    Manually register an existing model file with semantic versioning.
    Models are named: mcts_v1.0, mcts_v1.1, mcts_v1.2, etc.
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=400, detail=f"Model file not found: {file_path}")

    # Extract base name and get next semantic version
    base_name = name.lower().split("_v")[0] if "_v" in name.lower() else name.lower()
    version_str = get_next_version(db, base_name)
    major, minor = map(int, version_str.split("."))

    model = ModelDB(
        id=str(uuid.uuid4()),
        name=f"{base_name}_v{version_str}",  # e.g., "mcts_v1.0"
        version=major * 100 + minor,  # Store as integer for sorting (102 = v1.2)
        file_path=file_path,
        val_loss=val_loss,
        accuracy=accuracy,
        train_samples=train_samples,
        is_active=False,
        created_at=datetime.utcnow()
    )

    db.add(model)
    db.commit()
    db.refresh(model)

    return {
        "success": True,
        "data": {
            "id": model.id,
            "name": model.name,
            "version": f"v{version_str}",
            "filePath": model.file_path
        }
    }
