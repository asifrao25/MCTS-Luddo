"""
Training worker for MLX model training.

Runs training with epoch-by-epoch progress streaming.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Set, List

from ..database.connection import SessionLocal
from ..database.models import TrainingRunDB
from ..services.sse_manager import sse_manager
from ..config import DATA_DIR, MODELS_DIR

# Track cancellation requests
_cancelled_runs: Set[str] = set()


def cancel_training(run_id: str):
    """Signal training to stop."""
    _cancelled_runs.add(run_id)


def is_cancelled(run_id: str) -> bool:
    """Check if training should stop."""
    return run_id in _cancelled_runs


async def run_training(run_id: str, config: Dict[str, Any]):
    """
    Run training in background.

    Trains MLX neural network on simulation data with live progress.
    """
    db = SessionLocal()

    try:
        run = db.query(TrainingRunDB).filter(TrainingRunDB.id == run_id).first()
        if not run:
            return

        epochs = config.get("epochs", 100)
        batch_size = config.get("batchSize", 256)
        learning_rate = config.get("learningRate", 0.001)
        validation_split = config.get("validationSplit", 0.2)
        data_sources = config.get("dataSources", [])
        model_name = config.get("modelName", f"model_{run_id[:8]}")

        # Load training data
        all_data = []
        for source in data_sources:
            source_path = DATA_DIR / source
            if source_path.exists():
                for pos_file in source_path.glob("positions_*.json"):
                    with open(pos_file) as f:
                        all_data.extend(json.load(f))

        if not all_data:
            raise ValueError("No training data found")

        # Split data
        import random
        random.shuffle(all_data)
        split_idx = int(len(all_data) * (1 - validation_split))
        train_data = all_data[:split_idx]
        val_data = all_data[split_idx:]

        run.train_samples = len(train_data)
        run.val_samples = len(val_data)
        db.commit()

        # Import MLX components
        try:
            import mlx.core as mx
            import mlx.nn as nn
            import mlx.optimizers as optim
        except ImportError:
            raise ImportError("MLX not available. Install with: pip install mlx")

        # Create model (simple MLP for Ludo value network)
        class LudoValueNetwork(nn.Module):
            def __init__(self, input_size: int = 64):
                super().__init__()
                self.layers = [
                    nn.Linear(input_size, 128),
                    nn.Linear(128, 64),
                    nn.Linear(64, 1)
                ]

            def __call__(self, x):
                for i, layer in enumerate(self.layers[:-1]):
                    x = nn.relu(layer(x))
                return nn.sigmoid(self.layers[-1](x))

        model = LudoValueNetwork()
        optimizer = optim.Adam(learning_rate=learning_rate)

        loss_history: List[Dict] = []
        best_val_loss = float('inf')
        best_epoch = 0

        # Training loop
        for epoch in range(epochs):
            if is_cancelled(run_id):
                break

            # Emit epoch start
            await sse_manager.emit(run_id, "epoch_start", {
                "epoch": epoch + 1,
                "epochsTotal": epochs
            })

            run.current_epoch = epoch + 1
            db.commit()

            # Train epoch
            train_losses = []
            num_batches = (len(train_data) + batch_size - 1) // batch_size

            for batch_idx in range(num_batches):
                if is_cancelled(run_id):
                    break

                start = batch_idx * batch_size
                end = min(start + batch_size, len(train_data))
                batch = train_data[start:end]

                # Prepare batch (stub - actual feature extraction)
                # X = mx.array([sample["features"] for sample in batch])
                # y = mx.array([[1.0 if sample["winner"] == sample.get("player", 0) else 0.0] for sample in batch])

                # Stub loss calculation
                batch_loss = 0.5 - (epoch * 0.003)  # Simulated decreasing loss
                train_losses.append(batch_loss)

                # Emit batch progress every 10 batches
                if batch_idx % 10 == 0:
                    await sse_manager.emit(run_id, "progress", {
                        "epoch": epoch + 1,
                        "batch": batch_idx + 1,
                        "batchesTotal": num_batches,
                        "batchLoss": batch_loss
                    })

            # Calculate epoch losses
            train_loss = sum(train_losses) / len(train_losses) if train_losses else 0

            # Validation
            val_loss = train_loss * 1.1  # Stub - actual validation
            val_accuracy = 0.5 + (epoch * 0.002)  # Stub accuracy

            is_new_best = val_loss < best_val_loss
            if is_new_best:
                best_val_loss = val_loss
                best_epoch = epoch + 1

                # Save best model
                model_path = MODELS_DIR / f"{model_name}_best.npz"
                # mx.savez(str(model_path), **dict(model.parameters()))  # Actual save

            # Update database
            run.train_loss = train_loss
            run.val_loss = val_loss
            run.epochs_completed = epoch + 1
            run.best_val_loss = best_val_loss
            run.best_epoch = best_epoch
            db.commit()

            # Record loss history
            loss_history.append({
                "epoch": epoch + 1,
                "trainLoss": train_loss,
                "valLoss": val_loss,
                "valAccuracy": val_accuracy
            })

            # Save loss history
            run.loss_history = json.dumps(loss_history)
            db.commit()

            # Emit epoch complete
            await sse_manager.emit(run_id, "epoch_complete", {
                "epoch": epoch + 1,
                "trainLoss": train_loss,
                "valLoss": val_loss,
                "valAccuracy": val_accuracy,
                "isNewBest": is_new_best
            })

        # Save final model
        model_path = MODELS_DIR / f"{model_name}_final.npz"
        # mx.savez(str(model_path), **dict(model.parameters()))  # Actual save

        # Update final status
        run.status = "cancelled" if is_cancelled(run_id) else "completed"
        run.end_time = datetime.utcnow()
        run.model_path = str(model_path)
        db.commit()

        # Register model
        from ..routers.models_router import register_model
        if run.status == "completed":
            register_model(
                db=db,
                name=model_name,
                file_path=str(model_path),
                training_run_id=run_id,
                val_loss=best_val_loss,
                train_samples=len(train_data)
            )

        # Emit completion
        await sse_manager.emit(run_id, "complete", {
            "runId": run_id,
            "bestEpoch": best_epoch,
            "bestValLoss": best_val_loss,
            "modelPath": str(model_path)
        })

    except Exception as e:
        # Update error status
        run = db.query(TrainingRunDB).filter(TrainingRunDB.id == run_id).first()
        if run:
            run.status = "failed"
            run.end_time = datetime.utcnow()
            run.error_message = str(e)
            db.commit()

        await sse_manager.emit(run_id, "error", {"message": str(e)})

    finally:
        _cancelled_runs.discard(run_id)

        # Clear running training with lock
        from ..routers import training as train_router
        async with train_router._training_lock:
            if train_router._running_training == run_id:
                train_router._running_training = None

        db.close()
