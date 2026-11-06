"""
Maintenance Priority Engine: Evaluates operational urgency and assigns multi-tier priorities
using configurable, dataset-specific rules and resolving conflicting model evidence.
"""

from typing import Dict, Any, Tuple, List, Optional
from .recommendation_schema import RecommendationInput


class PriorityEngine:
    """
    Evaluates model outputs and assigns priority tiers:
    - Informational (Code 0)
    - Low priority (Code 1)
    - Medium priority (Code 2)
    - High priority (Code 3)
    - Critical review required (Code 4)

    Thresholds are strictly configurable and identified as experimental domain heuristics.
    NASA Remaining Useful Life and AI4I Failure Probability are treated as independent, non-interchangeable quantities.
    """

    PRIORITY_LEVELS = [
        "Informational",
        "Low priority",
        "Medium priority",
        "High priority",
        "Critical review required"
    ]

    DEFAULT_NASA_CONFIG = {
        "critical_rul_cycles": 20.0,
        "warning_rul_cycles": 50.0,
        "advisory_rul_cycles": 100.0,
        "early_warning_risk_high": 0.80,
    }

    DEFAULT_AI4I_CONFIG = {
        "critical_prob": 0.75,
        "warning_prob": 0.40,
        "advisory_prob": 0.20,
    }

    def __init__(
        self,
        nasa_config: Optional[Dict[str, float]] = None,
        ai4i_config: Optional[Dict[str, float]] = None
    ):
        self.nasa_config = dict(self.DEFAULT_NASA_CONFIG)
        if nasa_config:
            self.nasa_config.update(nasa_config)

        self.ai4i_config = dict(self.DEFAULT_AI4I_CONFIG)
        if ai4i_config:
            self.ai4i_config.update(ai4i_config)

    def evaluate_priority(self, payload: RecommendationInput) -> Tuple[str, str, List[str], bool]:
        """
        Evaluates input payload and returns:
            priority: str
            risk_status: str
            reasons: List[str]
            human_approval_required: bool
        """
        reasons: List[str] = []
        priority = "Informational"
        risk_status = "nominal"
        human_approval_required = False

        if payload.dataset_type == "NASA":
            rul = payload.predicted_rul
            severity = payload.anomaly_severity
            ew_risks = payload.early_warning_horizons

            # Check for conflict: High RUL but High Anomaly
            if rul is not None and rul > self.nasa_config["advisory_rul_cycles"] and severity == "High anomaly":
                priority = "High priority"
                risk_status = "conflicting_evidence"
                human_approval_required = True
                reasons.append(
                    f"Conflicting Evidence: Supervised RUL is high ({rul:.1f} cycles), "
                    f"yet unsupervised sensor detector flagged '{severity}' ({payload.anomaly_score:.4f}). "
                    f"Possible sensor fault or unmodeled operating regime."
                )
            elif rul is not None and rul <= self.nasa_config["critical_rul_cycles"]:
                priority = "Critical review required"
                risk_status = "imminent_exhaustion"
                human_approval_required = True
                reasons.append(
                    f"Predicted Remaining Useful Life is critically low ({rul:.1f} cycles <= {self.nasa_config['critical_rul_cycles']:.0f} cycles)."
                )
                if severity in ["Medium anomaly", "High anomaly"]:
                    reasons.append(f"Confirmed by acute multivariate sensor anomaly ({severity}).")

            elif rul is not None and rul <= self.nasa_config["warning_rul_cycles"]:
                priority = "High priority" if severity in ["Medium anomaly", "High anomaly"] else "Medium priority"
                risk_status = "accelerated_degradation"
                human_approval_required = True
                reasons.append(
                    f"Predicted RUL ({rul:.1f} cycles) has entered the degradation threshold (<= {self.nasa_config['warning_rul_cycles']:.0f} cycles)."
                )

            elif rul is not None and rul <= self.nasa_config["advisory_rul_cycles"]:
                if severity == "High anomaly":
                    priority = "High priority"
                    human_approval_required = True
                elif severity == "Medium anomaly":
                    priority = "Medium priority"
                    human_approval_required = True
                else:
                    priority = "Low priority"
                    human_approval_required = False
                risk_status = "incipient_wear"
                reasons.append(
                    f"Predicted RUL ({rul:.1f} cycles) indicates gradual wear progression."
                )

            elif severity == "High anomaly":
                priority = "High priority"
                risk_status = "sensor_anomaly"
                human_approval_required = True
                reasons.append(f"Unsupervised sensor detector flagged '{severity}'.")

            elif severity == "Medium anomaly":
                priority = "Medium priority"
                risk_status = "sensor_drift"
                human_approval_required = True
                reasons.append(f"Elevated sensor anomaly detected ({severity}).")

            elif severity == "Low anomaly":
                priority = "Low priority"
                risk_status = "minor_drift"
                human_approval_required = False
                reasons.append("Mild sensor baseline drift observed.")

            else:
                priority = "Informational"
                risk_status = "nominal"
                human_approval_required = False
                rul_text = f" (Predicted RUL: {rul:.1f} cycles)" if rul is not None else ""
                reasons.append(f"Engine operating within nominal envelope{rul_text}.")

            # Early warning horizon check
            if ew_risks and ew_risks.get("risk_within_10_cycles", 0.0) >= self.nasa_config["early_warning_risk_high"]:
                if priority != "Critical review required":
                    priority = "High priority"
                reasons.append(f"10-cycle early-warning classifier indicates {ew_risks['risk_within_10_cycles']*100:.1f}% failure probability.")

        # AI4I Workflow
        else:
            prob = payload.failure_risk_score
            severity = payload.anomaly_severity

            # Conflict check: Low Prob but High Anomaly
            if prob is not None and prob < self.ai4i_config["advisory_prob"] and severity == "High anomaly":
                priority = "High priority"
                risk_status = "conflicting_evidence"
                human_approval_required = True
                reasons.append(
                    f"Conflicting Evidence: Machine failure probability is low ({prob*100:.1f}%), "
                    f"yet operational condition detector flagged '{severity}'. "
                    f"Unusual operating load or sensor drift detected without immediate structural failure."
                )
            elif prob is not None and prob >= self.ai4i_config["critical_prob"]:
                priority = "Critical review required"
                risk_status = "imminent_breakdown"
                human_approval_required = True
                reasons.append(
                    f"Machine breakdown probability is critical ({prob*100:.1f}% >= {self.ai4i_config['critical_prob']*100:.0f}%)."
                )

            elif prob is not None and prob >= self.ai4i_config["warning_prob"]:
                priority = "High priority" if severity in ["Medium anomaly", "High anomaly"] else "Medium priority"
                risk_status = "elevated_risk"
                human_approval_required = True
                reasons.append(
                    f"Failure probability is elevated ({prob*100:.1f}% >= {self.ai4i_config['warning_prob']*100:.0f}%)."
                )

            elif prob is not None and prob >= self.ai4i_config["advisory_prob"]:
                priority = "Low priority"
                risk_status = "monitoring_recommended"
                human_approval_required = False
                reasons.append(
                    f"Mild failure risk detected ({prob*100:.1f}%)."
                )

            elif severity == "High anomaly":
                priority = "High priority"
                risk_status = "process_anomaly"
                human_approval_required = True
                reasons.append(f"Operational anomaly status: '{severity}'.")

            elif severity == "Medium anomaly":
                priority = "Medium priority"
                risk_status = "process_drift"
                human_approval_required = True
                reasons.append(f"Operational anomaly status: '{severity}'.")

            else:
                priority = "Informational"
                risk_status = "nominal"
                human_approval_required = False
                prob_text = f" (Breakdown Risk: {prob*100:.1f}%)" if prob is not None else ""
                reasons.append(f"Machine operating within normal tolerances{prob_text}.")

        return priority, risk_status, reasons, human_approval_required
