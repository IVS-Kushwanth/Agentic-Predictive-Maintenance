"""LSTM Neural Network Architectures for Predictive Maintenance.

Supports Remaining Useful Life (RUL) regression and temporal sequence modeling.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class LSTMRULModel(nn.Module):
    """Stacked Long Short-Term Memory (LSTM) network for RUL regression.
    
    Architecture:
    Input Layer: (batch_size, seq_len, input_dim)
    -> LSTM Layer 1 & 2
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

        self.lstm = nn.LSTM(
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
        lstm_out, _ = self.lstm(x)
        # Extract last time step: (batch_size, hidden_dim)
        last_step = lstm_out[:, -1, :]
        out = self.dropout(last_step)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc_out(out)
        return out.squeeze(-1)

    def get_config(self) -> Dict[str, Any]:
        """Return architecture configuration for persistence."""
        return {
            "model_type": "LSTM",
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout_rate,
            "fc_dim": self.fc_dim,
        }
