"""
Agent Orchestration Service for Phase 8 FastAPI Backend.
Executes multi-agent workflows using compiled LangGraph StateGraph:
Supervisor -> Prediction -> Anomaly -> Explanation -> Maintenance -> Result Validation.
"""

from typing import Dict, Any, List, Optional, Literal
import numpy as np
from src.graph.workflow import analyze_machine


class AgentService:
    """Coordinates multi-agent decision support workflows."""

    def analyze(
        self,
        machine_id: str,
        dataset_type: Literal["NASA", "AI4I"],
        sensor_data: List[float],
        workflow_type: str = "full_analysis"
    ) -> Dict[str, Any]:
        arr = np.asarray(sensor_data, dtype=np.float32)

        if dataset_type == "NASA":
            if arr.ndim == 1 and len(arr) != 153:
                raise ValueError(f"NASA C-MAPSS input requires 153 features, received {len(arr)}")
        else:
            if len(arr) != 14:
                raise ValueError(f"UCI AI4I input requires 14 features, received {len(arr)}")

        # Execute LangGraph workflow
        result = analyze_machine(
            input_data=arr,
            dataset_type=dataset_type,
            machine_id=machine_id,
            workflow_type=workflow_type
        )

        return result
