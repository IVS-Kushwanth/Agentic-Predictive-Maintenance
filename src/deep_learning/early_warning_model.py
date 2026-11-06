"""Temporal Early-Warning Classification Architectures for Predictive Maintenance.

Predicts failure probabilities for early-warning horizons (H = 10, 20, 30 cycles)
using recurrent backbones (LSTM or GRU) with class imbalance weighting.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class TemporalEarlyWarningClassifier(nn.Module):
    """Temporal Recurrent Classifier for early failure warning.
    
    Backbone: LSTM or GRU
    Output: Raw logit for binary classification (P(failure <= H cycles)).
    """

    def __init__(
        self,
        input_dim: int,
        backbone: str = "lstm",  # 'lstm' or 'gru'
        hidden_dim: int = 48,
        num_layers: int = 2,
        dropout: float = 0.2,
        fc_dim: int = 24,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.backbone_name = backbone.lower()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.fc_dim = fc_dim

        if self.backbone_name == "lstm":
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        elif self.backbone_name == "gru":
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose 'lstm' or 'gru'.")

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, fc_dim)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(fc_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, input_dim)
        rnn_out, _ = self.rnn(x)
        last_step = rnn_out[:, -1, :]
        out = self.dropout(last_step)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        logits = self.fc_out(out)
        return logits.squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute sigmoid failure probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

    def get_config(self) -> Dict[str, Any]:
        """Return architecture configuration for persistence."""
        return {
            "model_type": f"EarlyWarning_{self.backbone_name.upper()}",
            "backbone": self.backbone_name,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout_rate,
            "fc_dim": self.fc_dim,
        }
