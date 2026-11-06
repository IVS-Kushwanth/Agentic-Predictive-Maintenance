"""Gated Recurrent Unit (GRU) Neural Network Architectures for Predictive Maintenance.

Supports Remaining Useful Life (RUL) regression with lower parameter footprint and faster training.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class GRURULModel(nn.Module):
    """Stacked Gated Recurrent Unit (GRU) network for RUL regression.
    
    Architecture:
    Input Layer: (batch_size, seq_len, input_dim)
    -> GRU Layer 1 & 2
    -> Dropout
    -> Fully Connected Layer (Dense + ReLU)
    -> Output Linear Layer (1 unit -> predicted RUL)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        fc_dim: int = 32,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.fc_dim = fc_dim

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, fc_dim)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(fc_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        gru_out, _ = self.gru(x)
        # Extract last time step: (batch_size, hidden_dim)
        last_step = gru_out[:, -1, :]
        out = self.dropout(last_step)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc_out(out)
        return out.squeeze(-1)

    def get_config(self) -> Dict[str, Any]:
        """Return architecture configuration for persistence."""
        return {
            "model_type": "GRU",
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout_rate,
            "fc_dim": self.fc_dim,
        }
