"""
Deep Learning Explainability for recurrent neural network models (LSTM, GRU) using Integrated Gradients.
Computes 2D attribution tensors across time steps and sensor features.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients

from .schema import format_explanation_response


class DeepTemporalExplainer:
    """
    Computes Integrated Gradients attribution for PyTorch temporal sequence models (LSTMRULModel, GRURULModel).
    """

    def __init__(
        self,
        model: nn.Module,
        feature_names: List[str],
        baseline_sequence: Optional[np.ndarray] = None,
        model_id: str = "lstm_model",
        device: str = "cpu"
    ):
        self.model = model
        self.feature_names = feature_names
        self.model_id = model_id
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        self.ig = IntegratedGradients(self.model)
        self.baseline_tensor = (
            torch.tensor(baseline_sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
            if baseline_sequence is not None
            else None
        )

    def attribute_sequence(
        self,
        sequence: Union[np.ndarray, torch.Tensor],
        n_steps: int = 25
    ) -> Tuple[np.ndarray, float]:
        """
        Compute Integrated Gradients attribution for a single sequence (seq_len, features).
        Returns:
            attribution_matrix: (seq_len, features)
            predicted_rul: float
        """
        if isinstance(sequence, np.ndarray):
            tensor_x = torch.tensor(sequence, dtype=torch.float32).to(self.device)
        else:
            tensor_x = sequence.to(self.device)

        if tensor_x.ndim == 2:
            tensor_x = tensor_x.unsqueeze(0)  # Shape (1, L, D)

        with torch.no_grad():
            pred = float(self.model(tensor_x).item())

        baseline = self.baseline_tensor if self.baseline_tensor is not None else torch.zeros_like(tensor_x)

        attributions = self.ig.attribute(
            tensor_x,
            baselines=baseline,
            n_steps=n_steps
        )

        attr_matrix = attributions.squeeze(0).detach().cpu().numpy()  # (L, D)
        return attr_matrix, pred

    def explain_sequence(
        self,
        sequence: Union[np.ndarray, torch.Tensor],
        top_k_sensors: int = 5,
        n_steps: int = 25
    ) -> Dict[str, Any]:
        """
        Produces standardized explanation for a temporal sequence including sensor ranking and time-step weights.
        """
        attr_matrix, pred = self.attribute_sequence(sequence, n_steps=n_steps)
        seq_len, num_features = attr_matrix.shape

        # 1. Sensor-level attribution: sum of signed/absolute attribution across all time steps
        sensor_signed_contrib = np.sum(attr_matrix, axis=0)  # (D,)
        raw_last_timestep = sequence[-1] if isinstance(sequence, np.ndarray) else sequence.squeeze(0)[-1].cpu().numpy()

        # 2. Time-step attribution: sum of absolute attribution across all sensors
        time_step_weights = np.sum(np.abs(attr_matrix), axis=1)  # (L,)
        total_weight = float(np.sum(time_step_weights))
        if total_weight > 1e-9:
            norm_time_weights = [round(float(w / total_weight), 4) for w in time_step_weights]
        else:
            norm_time_weights = [round(1.0 / seq_len, 4)] * seq_len

        peak_step = int(np.argmax(time_step_weights)) + 1  # 1-indexed

        temporal_profile = {
            "window_length": seq_len,
            "peak_influence_cycle_in_window": peak_step,
            "recent_vs_early_weight_ratio": round(
                float(np.sum(time_step_weights[-10:]) / max(1e-9, np.sum(time_step_weights[:10]))), 2
            ),
            "time_step_weights": norm_time_weights
        }

        return format_explanation_response(
            prediction_val=pred,
            model_identifier=self.model_id,
            model_type=self.model.__class__.__name__,
            feature_names=self.feature_names,
            contributions=sensor_signed_contrib,
            raw_values=raw_last_timestep,
            task_type="regression",
            temporal_profile=temporal_profile,
            top_k=top_k_sensors
        )

    def compute_dataset_temporal_decay(
        self,
        sequences: np.ndarray,
        sample_size: int = 50,
        n_steps: int = 20
    ) -> np.ndarray:
        """
        Averages time-step attribution profiles across multiple sequences to evaluate historical decay.
        Returns 1D array of shape (L,).
        """
        n_seq = min(len(sequences), sample_size)
        time_profiles = []

        for i in range(n_seq):
            attr, _ = self.attribute_sequence(sequences[i], n_steps=n_steps)
            step_attr = np.sum(np.abs(attr), axis=1)
            total = np.sum(step_attr)
            if total > 1e-9:
                time_profiles.append(step_attr / total)

        return np.mean(time_profiles, axis=0)
