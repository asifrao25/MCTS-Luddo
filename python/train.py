#!/usr/bin/env python3
"""
MLX Neural Evaluator Training for Luddo Game

Trains a neural network to evaluate board positions using
self-play training data. Uses Apple MLX for efficient M-series training.
"""

import os
import json
import glob
from datetime import datetime
from typing import Tuple, Optional
import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("MLX not available. Install with: pip install mlx")

from tqdm import tqdm


class LuddoEvaluator(nn.Module):
    """
    Neural network for position evaluation.

    Architecture: 3-layer MLP (64 -> 128 -> 64 -> 1)
    Input: 64 features (4 players x 4 tokens x 4 features)
    Output: Position value (0-1, probability of winning)
    """

    def __init__(self, input_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.layer3 = nn.Linear(hidden_dim // 2, 1)

    def __call__(self, x):
        x = nn.relu(self.layer1(x))
        x = nn.relu(self.layer2(x))
        x = mx.sigmoid(self.layer3(x))
        return x


def load_training_data(data_dir: str) -> Tuple[mx.array, mx.array]:
    """Load training data from generated files"""
    # Find latest data files
    feature_files = sorted(glob.glob(os.path.join(data_dir, "features_*.npy")))
    outcome_files = sorted(glob.glob(os.path.join(data_dir, "outcomes_*.npy")))

    if not feature_files or not outcome_files:
        raise FileNotFoundError(f"No training data found in {data_dir}")

    print(f"Loading data from: {feature_files[-1]}")

    features = np.load(feature_files[-1])
    outcomes = np.load(outcome_files[-1])

    # Shuffle data
    indices = np.random.permutation(len(features))
    features = features[indices]
    outcomes = outcomes[indices]

    # Convert to MLX arrays
    features_mx = mx.array(features)
    outcomes_mx = mx.array(outcomes.reshape(-1, 1))

    print(f"Loaded {len(features)} training samples")
    return features_mx, outcomes_mx


def loss_fn(model, features, targets):
    """Binary cross-entropy loss"""
    predictions = model(features)
    # BCE loss: -[y*log(p) + (1-y)*log(1-p)]
    eps = 1e-7
    predictions = mx.clip(predictions, eps, 1 - eps)
    loss = -mx.mean(targets * mx.log(predictions) + (1 - targets) * mx.log(1 - predictions))
    return loss


def train_epoch(
    model: LuddoEvaluator,
    optimizer: optim.Optimizer,
    features: mx.array,
    targets: mx.array,
    batch_size: int = 64
) -> float:
    """Train for one epoch"""
    num_samples = features.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size
    total_loss = 0.0

    # Create loss and gradient function
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, num_samples)

        batch_features = features[start:end]
        batch_targets = targets[start:end]

        loss, grads = loss_and_grad_fn(model, batch_features, batch_targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        total_loss += loss.item()

    return total_loss / num_batches


def evaluate(model: LuddoEvaluator, features: mx.array, targets: mx.array) -> Tuple[float, float]:
    """Evaluate model on validation set"""
    predictions = model(features)
    loss = loss_fn(model, features, targets)

    # Calculate accuracy (threshold at 0.5)
    pred_binary = (predictions > 0.5).astype(mx.float32)
    accuracy = mx.mean((pred_binary == targets).astype(mx.float32))

    return loss.item(), accuracy.item()


def train(
    data_dir: str,
    output_dir: str,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    validation_split: float = 0.1
):
    """Train the neural evaluator"""
    if not HAS_MLX:
        print("MLX is required for training. Install with: pip install mlx")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    features, targets = load_training_data(data_dir)

    # Split into train/val
    num_samples = features.shape[0]
    num_val = int(num_samples * validation_split)
    num_train = num_samples - num_val

    train_features = features[:num_train]
    train_targets = targets[:num_train]
    val_features = features[num_train:]
    val_targets = targets[num_train:]

    print(f"Training samples: {num_train}")
    print(f"Validation samples: {num_val}")

    # Create model and optimizer
    model = LuddoEvaluator()
    optimizer = optim.Adam(learning_rate=learning_rate)

    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0

    print("\nStarting training...")
    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, optimizer, train_features, train_targets, batch_size)

        # Evaluate
        val_loss, val_accuracy = evaluate(model, val_features, val_targets)

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            save_model(model, output_dir, "best_model")

    print(f"\nBest model at epoch {best_epoch} with val loss: {best_val_loss:.4f}")

    # Save final model
    save_model(model, output_dir, "final_model")

    # Save training metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'num_train_samples': num_train,
        'num_val_samples': num_val
    }

    with open(os.path.join(output_dir, "training_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel saved to: {output_dir}")


def save_model(model: LuddoEvaluator, output_dir: str, name: str):
    """Save model weights"""
    # Flatten nested parameters into a flat dict
    flat_weights = {}
    for k, v in model.parameters().items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                flat_weights[f"{k}.{k2}"] = np.array(v2)
        else:
            flat_weights[k] = np.array(v)

    np.savez(os.path.join(output_dir, f"{name}.npz"), **flat_weights)


def load_model(model_path: str) -> LuddoEvaluator:
    """Load model from file"""
    model = LuddoEvaluator()

    # Load flat weights and reconstruct nested structure
    flat_weights = dict(np.load(model_path))
    nested_weights = {}

    for k, v in flat_weights.items():
        parts = k.split('.')
        if len(parts) == 2:
            layer, param = parts
            if layer not in nested_weights:
                nested_weights[layer] = {}
            nested_weights[layer][param] = mx.array(v)
        else:
            nested_weights[k] = mx.array(v)

    model.update(nested_weights)
    return model


def test_model(model_path: str, data_dir: str):
    """Test a trained model"""
    if not HAS_MLX:
        print("MLX is required. Install with: pip install mlx")
        return

    model = load_model(model_path)
    features, targets = load_training_data(data_dir)

    loss, accuracy = evaluate(model, features, targets)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Luddo Neural Evaluator")
    parser.add_argument("--data", type=str, default="../data/self_play_games", help="Training data directory")
    parser.add_argument("--output", type=str, default="../models", help="Output directory for model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--test", type=str, default=None, help="Path to model to test")

    args = parser.parse_args()

    if args.test:
        test_model(args.test, args.data)
    else:
        train(args.data, args.output, args.epochs, args.batch_size, args.lr)
