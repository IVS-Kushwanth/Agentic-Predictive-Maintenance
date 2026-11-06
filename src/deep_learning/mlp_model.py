"""Multilayer Perceptron (MLP) Classifier for UCI AI4I 2020 Tabular Predictive Maintenance.

Features batch normalization, dropout regularization, and pos_weight support
for severe class imbalance.
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn


class AI4IMLPClassifier(nn.Module):
    """Deep Tabular Multilayer Perceptron for Machine Failure Classification."""

    def __init__(
        self,
        input_dim: int = 14,
        hidden_dims: Tuple[int, ...] = (64, 32, 16),
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.fc_out = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        logits = self.fc_out(h)
        return logits.squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute sigmoid failure probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

    def get_config(self) -> Dict[str, Any]:
        """Return architecture configuration for persistence."""
        return {
            "model_type": "AI4I_MLP",
            "input_dim": self.input_dim,
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout_rate,
        }
