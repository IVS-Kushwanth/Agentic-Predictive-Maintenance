"""Hybrid CNN-LSTM Neural Network Architecture for Predictive Maintenance.

Combines 1D temporal convolutions for local feature extraction across sensor channels
with stacked LSTM recurrent layers for long-term temporal degradation modeling.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class CNNLSTMRULModel(nn.Module):
    """Hybrid Conv1D-LSTM architecture for Remaining Useful Life regression.
    
    Architecture:
    Input: (batch_size, seq_len, input_dim)
    -> Transpose to (batch_size, input_dim, seq_len)
    -> Conv1D (kernel_size=3, padding=1) + BatchNorm1D + ReLU
    -> Conv1D (kernel_size=3, padding=1) + ReLU
    -> Transpose back to (batch_size, seq_len, conv_filters)
    -> LSTM Layer(s)
    -> Dropout
    -> Fully Connected Layer (Dense + ReLU)
    -> Output Linear Layer (1 unit -> predicted RUL)
    """

    def __init__(
        self,
        input_dim: int,
        conv_filters: int = 32,
        kernel_size: int = 3,
        lstm_hidden_dim: int = 32,
        lstm_layers: int = 1,
        dropout: float = 0.2,
        fc_dim: int = 32,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.conv_filters = conv_filters
        self.kernel_size = kernel_size
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        self.dropout_rate = dropout
        self.fc_dim = fc_dim

        # 1D Temporal Convolutions
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=conv_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.bn1 = nn.BatchNorm1d(conv_filters)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(
            in_channels=conv_filters,
            out_channels=conv_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.relu2 = nn.ReLU()

        # Recurrent temporal modeling
        self.lstm = nn.LSTM(
            input_size=conv_filters,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_hidden_dim, fc_dim)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(fc_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, input_dim)
        # Transpose for Conv1d: (batch_size, input_dim, seq_len)
        x_conv = x.transpose(1, 2)
        h = self.conv1(x_conv)
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.conv2(h)
        h = self.relu2(h)

        # Transpose back: (batch_size, seq_len, conv_filters)
        h_seq = h.transpose(1, 2)
        lstm_out, _ = self.lstm(h_seq)
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
            "model_type": "CNN-LSTM",
            "input_dim": self.input_dim,
            "conv_filters": self.conv_filters,
            "kernel_size": self.kernel_size,
            "lstm_hidden_dim": self.lstm_hidden_dim,
            "lstm_layers": self.lstm_layers,
            "dropout": self.dropout_rate,
            "fc_dim": self.fc_dim,
        }
