"""
Autoencoder architectures for Phase 4 Multivariate Sensor Anomaly Detection:
- LSTMAutoencoder: For 3D multivariate temporal sequences (NASA C-MAPSS).
- DenseAutoencoder: For 2D tabular sensor/operating condition data (AI4I).

Both compute sample reconstruction error (anomaly score) and feature-level errors for sensor attribution.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, Optional, Union, List, Tuple

from .base import BaseAnomalyDetector


# ---------------------------------------------------------
# PyTorch Neural Network Modules
# ---------------------------------------------------------

class PyTorchLSTMAutoencoder(nn.Module):
    """
    Encoder-Decoder LSTM Autoencoder for temporal multivariate sequences.
    Input: (Batch, Sequence_Length, Features)
    Reconstruction: (Batch, Sequence_Length, Features)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, latent_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder_lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.encoder_lstm2 = nn.LSTM(hidden_dim, latent_dim, batch_first=True)

        # Decoder
        self.decoder_lstm1 = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Encode: x -> hidden -> latent vector
        out1, _ = self.encoder_lstm1(x)
        _, (h_n, _) = self.encoder_lstm2(out1)
        latent = h_n[-1]  # Shape: (batch_size, latent_dim)

        # Repeat latent vector across sequence length
        repeated_latent = latent.unsqueeze(1).repeat(1, seq_len, 1)  # (batch_size, seq_len, latent_dim)

        # Decode
        dec_out, _ = self.decoder_lstm1(repeated_latent)
        reconstruction = self.output_layer(dec_out)  # (batch_size, seq_len, input_dim)

        return reconstruction


class PyTorchDenseAutoencoder(nn.Module):
    """
    Standard Feedforward Autoencoder for 2D tabular features.
    Input: (Batch, Features)
    Reconstruction: (Batch, Features)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32, latent_dim: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


# ---------------------------------------------------------
# Detector Wrapper Classes (BaseAnomalyDetector)
# ---------------------------------------------------------

class LSTMAutoencoderDetector(BaseAnomalyDetector):
    """
    Detector wrapping the PyTorch LSTM Autoencoder for temporal sequences.
    Reconstruction error = Mean Squared Error between input sequence and reconstruction.
    """

    def __init__(
        self,
        model_id: str = "lstm_autoencoder",
        dataset_name: str = "FD001",
        input_dim: int = 153,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        lr: float = 1e-3,
        batch_size: int = 128,
        epochs: int = 20,
        device: str = "cpu"
    ):
        super().__init__(model_id=model_id, model_type="LSTMAutoencoder", dataset_name=dataset_name)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device(device)

        self.nn_model: Optional[PyTorchLSTMAutoencoder] = None
        self.training_history: List[float] = []

    def fit(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        val_X: Optional[np.ndarray] = None,
        **kwargs
    ) -> "LSTMAutoencoderDetector":
        """
        Fit LSTM Autoencoder on nominal sequences X of shape (N, L, D).
        """
        if X.ndim != 3:
            raise ValueError(f"LSTM Autoencoder requires 3D sequences (N, L, D), got shape {X.shape}")

        _, _, D = X.shape
        self.input_dim = D
        if feature_names:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"sensor_{i}" for i in range(D)]

        self.nn_model = PyTorchLSTMAutoencoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim
        ).to(self.device)

        optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.nn_model.train()
        self.training_history = []

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            total_samples = 0
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                recon = self.nn_model(batch_x)
                loss = criterion(recon, batch_x)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_x.size(0)
                total_samples += batch_x.size(0)

            avg_loss = epoch_loss / max(1, total_samples)
            self.training_history.append(avg_loss)

        self.is_fitted = True
        return self

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Return reconstructed sequence of shape (N, L, D)."""
        if not self.is_fitted or self.nn_model is None:
            raise RuntimeError(f"{self.model_id} must be fitted before reconstruction.")
        if X.ndim != 3:
            raise ValueError(f"Expected 3D array (N, L, D), got shape {X.shape}")

        self.nn_model.eval()
        with torch.no_grad():
            tensor_x = torch.tensor(X, dtype=torch.float32).to(self.device)
            # Process in batches to conserve memory
            reconstructions = []
            for i in range(0, len(tensor_x), self.batch_size):
                batch = tensor_x[i:i+self.batch_size]
                recon = self.nn_model(batch)
                reconstructions.append(recon.cpu().numpy())
            return np.concatenate(reconstructions, axis=0)

    def feature_errors(self, X: np.ndarray) -> np.ndarray:
        """
        Compute per-feature reconstruction MSE averaged across the sequence window.
        Returns array of shape (N, D).
        """
        recon = self.reconstruct(X)
        # Squared error per element: (N, L, D)
        sq_err = (X - recon) ** 2
        # Average across sequence length L -> (N, D)
        return np.mean(sq_err, axis=1)

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute overall anomaly score per sequence (mean reconstruction MSE across all features).
        Returns 1D array of shape (N,).
        """
        feat_err = self.feature_errors(X)
        return np.mean(feat_err, axis=1)

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "dataset_name": self.dataset_name,
            "thresholds": self.thresholds,
            "feature_names": self.feature_names,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "training_history": self.training_history,
            "metadata": self.metadata,
            "state_dict": self.nn_model.state_dict() if self.nn_model else None
        }
        torch.save(checkpoint, filepath)

    def load(self, filepath: str) -> "LSTMAutoencoderDetector":
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model_id = checkpoint["model_id"]
        self.model_type = checkpoint["model_type"]
        self.dataset_name = checkpoint["dataset_name"]
        self.thresholds = checkpoint["thresholds"]
        self.feature_names = checkpoint["feature_names"]
        self.input_dim = checkpoint["input_dim"]
        self.hidden_dim = checkpoint["hidden_dim"]
        self.latent_dim = checkpoint["latent_dim"]
        self.training_history = checkpoint.get("training_history", [])
        self.metadata = checkpoint.get("metadata", {})

        self.nn_model = PyTorchLSTMAutoencoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim
        ).to(self.device)
        if checkpoint.get("state_dict"):
            self.nn_model.load_state_dict(checkpoint["state_dict"])
        self.is_fitted = True
        return self


class DenseAutoencoderDetector(BaseAnomalyDetector):
    """
    Detector wrapping standard Dense Autoencoder for 2D tabular features.
    Reconstruction error = Mean Squared Error between input vector and reconstruction.
    """

    def __init__(
        self,
        model_id: str = "dense_autoencoder",
        dataset_name: str = "AI4I",
        input_dim: int = 14,
        hidden_dim: int = 32,
        latent_dim: int = 8,
        lr: float = 1e-3,
        batch_size: int = 128,
        epochs: int = 25,
        device: str = "cpu"
    ):
        super().__init__(model_id=model_id, model_type="DenseAutoencoder", dataset_name=dataset_name)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device(device)

        self.nn_model: Optional[PyTorchDenseAutoencoder] = None
        self.training_history: List[float] = []

    def fit(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        **kwargs
    ) -> "DenseAutoencoderDetector":
        if X.ndim != 2:
            raise ValueError(f"Dense Autoencoder requires 2D array (N, D), got shape {X.shape}")

        N, D = X.shape
        self.input_dim = D
        if feature_names:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"feat_{i}" for i in range(D)]

        self.nn_model = PyTorchDenseAutoencoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim
        ).to(self.device)

        optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.nn_model.train()
        self.training_history = []

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            total_samples = 0
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                recon = self.nn_model(batch_x)
                loss = criterion(recon, batch_x)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_x.size(0)
                total_samples += batch_x.size(0)

            avg_loss = epoch_loss / max(1, total_samples)
            self.training_history.append(avg_loss)

        self.is_fitted = True
        return self

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.nn_model is None:
            raise RuntimeError(f"{self.model_id} must be fitted before reconstruction.")
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array (N, D), got shape {X.shape}")

        self.nn_model.eval()
        with torch.no_grad():
            tensor_x = torch.tensor(X, dtype=torch.float32).to(self.device)
            reconstructions = []
            for i in range(0, len(tensor_x), self.batch_size):
                batch = tensor_x[i:i+self.batch_size]
                recon = self.nn_model(batch)
                reconstructions.append(recon.cpu().numpy())
            return np.concatenate(reconstructions, axis=0)

    def feature_errors(self, X: np.ndarray) -> np.ndarray:
        """
        Compute per-feature squared error (N, D).
        """
        recon = self.reconstruct(X)
        return (X - recon) ** 2

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute overall anomaly score (mean squared reconstruction error).
        Returns shape (N,).
        """
        feat_err = self.feature_errors(X)
        return np.mean(feat_err, axis=1)

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "dataset_name": self.dataset_name,
            "thresholds": self.thresholds,
            "feature_names": self.feature_names,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "training_history": self.training_history,
            "metadata": self.metadata,
            "state_dict": self.nn_model.state_dict() if self.nn_model else None
        }
        torch.save(checkpoint, filepath)

    def load(self, filepath: str) -> "DenseAutoencoderDetector":
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model_id = checkpoint["model_id"]
        self.model_type = checkpoint["model_type"]
        self.dataset_name = checkpoint["dataset_name"]
        self.thresholds = checkpoint["thresholds"]
        self.feature_names = checkpoint["feature_names"]
        self.input_dim = checkpoint["input_dim"]
        self.hidden_dim = checkpoint["hidden_dim"]
        self.latent_dim = checkpoint["latent_dim"]
        self.training_history = checkpoint.get("training_history", [])
        self.metadata = checkpoint.get("metadata", {})

        self.nn_model = PyTorchDenseAutoencoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim
        ).to(self.device)
        if checkpoint.get("state_dict"):
            self.nn_model.load_state_dict(checkpoint["state_dict"])
        self.is_fitted = True
        return self
