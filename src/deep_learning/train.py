"""Training Engine for Deep Learning Predictive Maintenance Models.

Optimized for CPU-based high-throughput execution with:
- Reproducible random seeds.
- Adam optimization with weight decay.
- Learning rate scheduling with ReduceLROnPlateau.
- Early stopping on validation loss.
- Best model state_dict checkpointing.
- Training and validation metric tracking across epochs.
"""

import copy
import random
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    """Early stops training when validation loss stops improving."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4, mode: str = "min") -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.best_state_dict: Optional[Dict[str, Any]] = None
        self.best_epoch = 0

    def step(self, current_score: float, model: nn.Module, epoch: int) -> bool:
        if self.mode == "min":
            score = -current_score
        else:
            score = current_score

        if self.best_score is None:
            self.best_score = score
            self.best_state_dict = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            return True
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_score = score
            self.best_state_dict = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            self.counter = 0
            return True


def train_rul_regressor(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 20,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    patience: int = 5,
    loss_fn_name: str = "smooth_l1",  # 'smooth_l1' (Huber) or 'mse'
    device: str = "cpu",
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train an RUL regression model with early stopping and learning rate scheduling."""
    set_seed(seed)
    model = model.to(device)

    if loss_fn_name in ["smooth_l1", "huber"]:
        criterion = nn.SmoothL1Loss(beta=1.0)
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5
    )
    early_stopper = EarlyStopping(patience=patience, mode="min")

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_rmse": [],
        "lr": [],
    }

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_batches = 0

        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            preds = model(X_b)
            loss = criterion(preds, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_total += loss.item()
            train_batches += 1

        train_epoch_loss = train_loss_total / max(train_batches, 1)

        # Validation
        model.eval()
        val_loss_total = 0.0
        val_mse_total = 0.0
        val_batches = 0

        with torch.no_grad():
            for X_vb, y_vb in val_loader:
                X_vb, y_vb = X_vb.to(device), y_vb.to(device)
                val_preds = model(X_vb)
                v_loss = criterion(val_preds, y_vb)
                val_loss_total += v_loss.item()
                val_mse_total += torch.mean((val_preds - y_vb) ** 2).item()
                val_batches += 1

        val_epoch_loss = val_loss_total / max(val_batches, 1)
        val_rmse = float(np.sqrt(val_mse_total / max(val_batches, 1)))
        curr_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(round(train_epoch_loss, 4))
        history["val_loss"].append(round(val_epoch_loss, 4))
        history["val_rmse"].append(round(val_rmse, 4))
        history["lr"].append(curr_lr)

        scheduler.step(val_epoch_loss)
        is_best = early_stopper.step(val_epoch_loss, model, epoch)

        if verbose and (epoch % 3 == 0 or epoch == 1 or is_best):
            best_flag = " *" if is_best else ""
            print(
                f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_epoch_loss:.4f} | "
                f"Val Loss: {val_epoch_loss:.4f} | Val RMSE: {val_rmse:.2f} | LR: {curr_lr:.6f}{best_flag}",
                flush=True
            )

        if early_stopper.early_stop:
            if verbose:
                print(f"Early stopping triggered at epoch {epoch}. Best epoch was {early_stopper.best_epoch}.", flush=True)
            break

    train_duration = time.time() - start_time

    if early_stopper.best_state_dict is not None:
        model.load_state_dict(early_stopper.best_state_dict)

    return {
        "best_epoch": early_stopper.best_epoch,
        "best_val_loss": round(float(-early_stopper.best_score), 4) if early_stopper.best_score else None,
        "train_duration_sec": round(train_duration, 2),
        "history": history,
    }


def train_binary_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    pos_weight: float = 1.0,
    epochs: int = 20,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    patience: int = 5,
    device: str = "cpu",
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train a binary classification model using BCEWithLogitsLoss."""
    set_seed(seed)
    model = model.to(device)

    pos_wt_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_wt_tensor)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5
    )
    early_stopper = EarlyStopping(patience=patience, mode="min")

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
    }

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_batches = 0

        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_total += loss.item()
            train_batches += 1

        train_epoch_loss = train_loss_total / max(train_batches, 1)

        # Validation
        model.eval()
        val_loss_total = 0.0
        val_batches = 0

        with torch.no_grad():
            for X_vb, y_vb in val_loader:
                X_vb, y_vb = X_vb.to(device), y_vb.to(device)
                logits = model(X_vb)
                v_loss = criterion(logits, y_vb)
                val_loss_total += v_loss.item()
                val_batches += 1

        val_epoch_loss = val_loss_total / max(val_batches, 1)
        curr_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(round(train_epoch_loss, 4))
        history["val_loss"].append(round(val_epoch_loss, 4))
        history["lr"].append(curr_lr)

        scheduler.step(val_epoch_loss)
        is_best = early_stopper.step(val_epoch_loss, model, epoch)

        if verbose and (epoch % 3 == 0 or epoch == 1 or is_best):
            best_flag = " *" if is_best else ""
            print(
                f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_epoch_loss:.4f} | "
                f"Val Loss: {val_epoch_loss:.4f} | LR: {curr_lr:.6f}{best_flag}",
                flush=True
            )

        if early_stopper.early_stop:
            if verbose:
                print(f"Early stopping triggered at epoch {epoch}. Best epoch was {early_stopper.best_epoch}.", flush=True)
            break

    train_duration = time.time() - start_time

    if early_stopper.best_state_dict is not None:
        model.load_state_dict(early_stopper.best_state_dict)

    return {
        "best_epoch": early_stopper.best_epoch,
        "best_val_loss": round(float(-early_stopper.best_score), 4) if early_stopper.best_score else None,
        "train_duration_sec": round(train_duration, 2),
        "history": history,
    }
