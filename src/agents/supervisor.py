"""
Supervisor Agent:
- Receives user request and input data
- Validates inputs (dimensions, non-null, dataset alignment)
- Determines required workflow route
- Coordinates shared state and handles errors
"""

import uuid
from typing import Dict, Any, List
import numpy as np

from src.graph.state import MaintenanceState


class SupervisorAgent:
    """
    Supervisor agent managing incoming requests, validation, and workflow state.
    """

    def __init__(self):
        pass

    def run(self, state: MaintenanceState) -> MaintenanceState:
        """
        Validates state and initializes execution.
        """
        new_state = dict(state)
        errors: List[str] = list(new_state.get("error_information", []))

        # Ensure request ID
        if not new_state.get("request_id"):
            new_state["request_id"] = str(uuid.uuid4())[:8]

        # Ensure machine ID
        if not new_state.get("machine_id"):
            new_state["machine_id"] = "UNIT-001"

        # Validate dataset_type
        ds_type = new_state.get("dataset_type")
        if ds_type not in ["NASA", "AI4I"]:
            errors.append(f"Invalid or missing dataset_type: '{ds_type}'. Expected 'NASA' or 'AI4I'.")

        # Validate input_sensor_data
        data = new_state.get("input_sensor_data")
        if data is None:
            errors.append("Missing input_sensor_data. Sensor readings must be provided.")
        else:
            try:
                arr = np.asarray(data, dtype=np.float32)
                if arr.size == 0:
                    errors.append("input_sensor_data is empty.")
                if np.isnan(arr).any():
                    errors.append("input_sensor_data contains NaN values.")

                # Dimension checks
                if ds_type == "AI4I":
                    if (arr.ndim == 1 and arr.shape[0] != 14) or (arr.ndim == 2 and arr.shape[-1] != 14):
                        errors.append(f"AI4I dataset requires exactly 14 operational features, received shape {arr.shape}.")
                elif ds_type == "NASA":
                    if (arr.ndim == 1 and arr.shape[0] != 153) and (arr.ndim == 2 and arr.shape[-1] != 153) and (arr.ndim == 3 and arr.shape[-1] != 153):
                        errors.append(f"NASA dataset requires 153 features, received shape {arr.shape}.")
            except Exception as e:
                errors.append(f"Failed to parse input_sensor_data as numerical array: {str(e)}")

        if errors:
            new_state["error_information"] = errors
            new_state["workflow_status"] = "FAILED"
        else:
            new_state["workflow_status"] = "VALIDATED"

        return new_state
