"""
Recommendation Rules: Maps model evidence, top contributing sensors, and priority tiers
into concrete, evidence-linked industrial maintenance suggestions.
"""

from typing import Dict, Any, List, Optional, Tuple
from .recommendation_schema import RecommendationInput


class RecommendationRules:
    """
    Expert rule engine mapping physical degradation signals to industrial maintenance protocols.
    Every recommendation is explicitly linked to supporting mathematical evidence.
    """

    NASA_SUBSYSTEM_MAP = {
        "sensor_11": ("HPC Thermal Assembly", "Perform borescope inspection of High-Pressure Compressor outlet; calibrate static thermocouple T30."),
        "sensor_4": ("LPT Exhaust Module", "Inspect Low-Pressure Turbine stator blades for thermal erosion and check exhaust thermocouple harness."),
        "sensor_12": ("Fan & Bypass Duct", "Inspect bypass duct aerodynamic seals and check fan blade tip clearance."),
        "sensor_13": ("Coolant Bleed System", "Test high-pressure core bleed valve actuation and inspect bleed line seals."),
        "sensor_14": ("Pressure Regulation", "Verify core coolant regulator differential pressure and check pressure transducer calibration."),
        "sensor_8": ("Fan Bearing Assembly", "Conduct vibration spectral analysis on fan shaft and inspect front bearing lubrication."),
        "sensor_9": ("Core Rotor Shaft", "Check High-Pressure core spool rotational balance and inspect mid-shaft bearings."),
        "sensor_15": ("Bypass Aerodynamics", "Clean bypass flow vanes and verify static pressure tap ports."),
        "sensor_7": ("HPC Diffuser Case", "Inspect HPC outlet diffuser casing for pressure seal integrity.")
    }

    AI4I_SUBSYSTEM_MAP = {
        "Mechanical power": ("Spindle Motor & Drive", "Inspect electrical drive inverter, motor winding resistance, and spindle drive belt tension."),
        "Tool wear": ("Cutting Tool Assembly", "Replace cutting tool insert and verify tool holder runout tolerance (< 0.005 mm)."),
        "Rotational speed": ("Spindle Bearings", "Perform high-speed spindle bearing acoustic emission test; verify drive spindle lubrication."),
        "Overstrain indicator": ("Workpiece Clamping & Feed", "Reduce axis feed rate by 15%; verify workpiece fixture clamping pressure and torque limits."),
        "Torque": ("Drive Gearbox", "Inspect drive train gearbox backlash and check torque sensor calibration."),
        "Torque to speed ratio": ("Mechanical Drive Train", "Inspect mechanical coupling for slippage or binding during load transitions."),
        "Heat dissipation index": ("Coolant Delivery System", "Flush coolant filtration lines, verify nozzle pressure (> 2.5 bar), and clean heat exchanger."),
        "Air temperature": ("Ambient Environmental Control", "Check electrical enclosure cooling unit and air circulation filters."),
        "Process temperature": ("Machining Zone Thermal Control", "Inspect direct workpiece coolant flooding and check thermal sensor calibration.")
    }

    def generate_recommendations(
        self,
        payload: RecommendationInput,
        priority: str,
        risk_status: str,
        reasons: List[str]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Generates actionable recommendations and supporting evidence lists.
        """
        actions: List[Dict[str, Any]] = []
        evidence: List[Dict[str, Any]] = []

        # 1. Compile Supporting Evidence
        if payload.dataset_type == "NASA":
            if payload.predicted_rul is not None:
                evidence.append({
                    "evidence_type": "Supervised Prediction",
                    "metric": "Remaining Useful Life",
                    "value": f"{payload.predicted_rul:.1f} cycles",
                    "status": "Degrading" if payload.predicted_rul <= 50.0 else "Nominal"
                })
            if payload.anomaly_severity:
                evidence.append({
                    "evidence_type": "Unsupervised Anomaly",
                    "metric": "Anomaly Severity",
                    "value": payload.anomaly_severity,
                    "score": payload.anomaly_score
                })
        else:
            if payload.failure_risk_score is not None:
                evidence.append({
                    "evidence_type": "Supervised Prediction",
                    "metric": "Failure Probability",
                    "value": f"{payload.failure_risk_score*100:.1f}%",
                    "status": "High Risk" if payload.failure_risk_score >= 0.50 else "Nominal"
                })
            if payload.anomaly_severity:
                evidence.append({
                    "evidence_type": "Unsupervised Anomaly",
                    "metric": "Anomaly Severity",
                    "value": payload.anomaly_severity,
                    "score": payload.anomaly_score
                })

        for s in payload.important_sensors[:3]:
            feat_name = s.get("feature", "unknown")
            imp_val = s.get("importance", 0.0)
            contrib_dir = s.get("contribution", "positive")
            evidence.append({
                "evidence_type": "Attribution Driver",
                "feature": feat_name,
                "importance": imp_val,
                "direction": contrib_dir
            })

        # 2. Formulate Specific Actions based on Evidence & Priority
        top_feature = payload.important_sensors[0]["feature"] if payload.important_sensors else ""

        if payload.dataset_type == "NASA":
            subsystem_name, protocol = self._match_nasa_subsystem(top_feature)

            if priority in ["Critical review required", "High priority"]:
                actions.append({
                    "action_id": "REC-NASA-01",
                    "priority": "HIGH",
                    "category": "INSPECTION_OVERHAUL",
                    "subsystem": subsystem_name,
                    "target_subsystem": subsystem_name,
                    "urgency": "IMMEDIATE (Within 2-5 Operating Cycles)",
                    "action_description": protocol,
                    "description": protocol,
                    "protocol_reference": "EASA / FAA Part 145 Engine Overhaul Standards",
                    "evidence_link": f"Triggered by acute degradation in '{top_feature}' (RUL: {payload.predicted_rul or 'N/A'} cycles, Severity: {payload.anomaly_severity})",
                    "justification": f"Triggered by acute degradation in '{top_feature}' (RUL: {payload.predicted_rul or 'N/A'} cycles, Severity: {payload.anomaly_severity})"
                })
                actions.append({
                    "action_id": "REC-NASA-02",
                    "priority": "HIGH",
                    "category": "LOGISTICS_CONTINGENCY",
                    "subsystem": "Fleet Hangar Support",
                    "target_subsystem": "Fleet Hangar Support",
                    "urgency": "IMMEDIATE",
                    "action_description": f"Pre-allocate standby turbofan engine at maintenance base for unit {payload.machine_id}.",
                    "description": f"Pre-allocate standby turbofan engine at maintenance base for unit {payload.machine_id}.",
                    "protocol_reference": "Standard Airline Line Replaceable Unit (LRU) Protocol",
                    "evidence_link": "High risk of unscheduled engine removal (UER)",
                    "justification": "High risk of unscheduled engine removal (UER)"
                })

            elif priority == "Medium priority":
                actions.append({
                    "action_id": "REC-NASA-03",
                    "priority": "MEDIUM",
                    "category": "TARGETED_DIAGNOSTIC",
                    "subsystem": subsystem_name,
                    "target_subsystem": subsystem_name,
                    "urgency": "PLANNED (Next Scheduled Line Check)",
                    "action_description": protocol,
                    "description": protocol,
                    "protocol_reference": "Turbofan Scheduled Maintenance Manual",
                    "evidence_link": f"Moderate wear detected in '{top_feature}'",
                    "justification": f"Moderate wear detected in '{top_feature}'"
                })
                actions.append({
                    "action_id": "REC-NASA-04",
                    "priority": "MEDIUM",
                    "category": "TELEMETRY_MONITORING",
                    "subsystem": "Flight Data Analytics",
                    "target_subsystem": "Flight Data Analytics",
                    "urgency": "CONTINUOUS",
                    "action_description": "Enable high-rate operational flight data monitoring for thermodynamic channel deviations.",
                    "description": "Enable high-rate operational flight data monitoring for thermodynamic channel deviations.",
                    "protocol_reference": "Condition Monitoring Program (CMP)",
                    "evidence_link": "Elevated sensor anomaly status",
                    "justification": "Elevated sensor anomaly status"
                })

            else:
                actions.append({
                    "action_id": "REC-NASA-05",
                    "priority": "LOW" if priority == "Low priority" else "INFO",
                    "category": "ROUTINE_MONITORING",
                    "subsystem": "Turbofan Assembly",
                    "target_subsystem": "Turbofan Assembly",
                    "urgency": "ROUTINE",
                    "action_description": f"Maintain standard flight hour inspection schedule. Current telemetry indicates nominal operating health.",
                    "description": f"Maintain standard flight hour inspection schedule. Current telemetry indicates nominal operating health.",
                    "protocol_reference": "Standard Maintenance Program",
                    "evidence_link": "Model metrics confirm engine operating within nominal design envelope",
                    "justification": "Model metrics confirm engine operating within nominal design envelope"
                })

        # AI4I Recommendations
        else:
            subsystem_name, protocol = self._match_ai4i_subsystem(top_feature)

            if priority in ["Critical review required", "High priority"]:
                actions.append({
                    "action_id": "REC-AI4I-01",
                    "priority": "HIGH",
                    "category": "PHYSICAL_INTERVENTION",
                    "subsystem": subsystem_name,
                    "target_subsystem": subsystem_name,
                    "urgency": "IMMEDIATE (Current Shift End)",
                    "action_description": protocol,
                    "description": protocol,
                    "protocol_reference": "CNC Machine Tool Safety & Maintenance ISO 10791",
                    "evidence_link": f"Triggered by high breakdown probability ({payload.failure_risk_score*100 if payload.failure_risk_score else 0:.1f}%) and excessive wear in '{top_feature}'",
                    "justification": f"Triggered by high breakdown probability ({payload.failure_risk_score*100 if payload.failure_risk_score else 0:.1f}%) and excessive wear in '{top_feature}'"
                })
                actions.append({
                    "action_id": "REC-AI4I-02",
                    "priority": "HIGH",
                    "category": "PROCESS_MODULATION",
                    "subsystem": "CNC Controller",
                    "target_subsystem": "CNC Controller",
                    "urgency": "IMMEDIATE",
                    "action_description": "Throttle spindle load by 20% and reduce feed rate to prevent catastrophic tool rupture.",
                    "description": "Throttle spindle load by 20% and reduce feed rate to prevent catastrophic tool rupture.",
                    "protocol_reference": "Machining Parameter Override Protocol",
                    "evidence_link": "Overstrain protection threshold exceeded",
                    "justification": "Overstrain protection threshold exceeded"
                })

            elif priority == "Medium priority":
                actions.append({
                    "action_id": "REC-AI4I-03",
                    "priority": "MEDIUM",
                    "category": "PREVENTIVE_MAINTENANCE",
                    "subsystem": subsystem_name,
                    "target_subsystem": subsystem_name,
                    "urgency": "PLANNED (Within 24 Hours)",
                    "action_description": protocol,
                    "description": protocol,
                    "protocol_reference": "Shop Floor Daily Preventive Protocol",
                    "evidence_link": f"Process telemetry shows anomalous deviation in '{top_feature}'",
                    "justification": f"Process telemetry shows anomalous deviation in '{top_feature}'"
                })

            else:
                actions.append({
                    "action_id": "REC-AI4I-04",
                    "priority": "LOW" if priority == "Low priority" else "INFO",
                    "category": "CONTINUOUS_MONITORING",
                    "subsystem": "Machining Center",
                    "target_subsystem": "Machining Center",
                    "urgency": "ROUTINE",
                    "action_description": "Continue standard production run; log tool wear progression and spindle thermal dissipation.",
                    "description": "Continue standard production run; log tool wear progression and spindle thermal dissipation.",
                    "protocol_reference": "Standard Operating Procedure (SOP)",
                    "evidence_link": "Operational features indicate low breakdown probability",
                    "justification": "Operational features indicate low breakdown probability"
                })

        return actions, evidence

    def _match_nasa_subsystem(self, feature_name: str) -> Tuple[str, str]:
        for k, (sub, desc) in self.NASA_SUBSYSTEM_MAP.items():
            if k in feature_name:
                return sub, desc
        return "Core Engine Assembly", "Conduct visual and electronic diagnostic check on engine core telemetry sensors."

    def _match_ai4i_subsystem(self, feature_name: str) -> Tuple[str, str]:
        for k, (sub, desc) in self.AI4I_SUBSYSTEM_MAP.items():
            if k in feature_name:
                return sub, desc
        return "Spindle & Tooling Drive", "Inspect mechanical alignment and verify operating parameters."
