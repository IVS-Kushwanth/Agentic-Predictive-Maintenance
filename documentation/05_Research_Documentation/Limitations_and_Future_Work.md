# Technical Limitations & Future Research Directions
## Agentic Predictive Maintenance Framework

In adherence to rigorous academic standards, this document transparently discusses the system's operational boundaries, dataset constraints, and avenues for future research.

---

## 1. Technical Limitations

### 1.1 Dataset & Simulation Constraints
- **Simulation Artifacts in C-MAPSS**: The NASA C-MAPSS benchmark is generated from a thermodynamic computer simulation (C-MAPSS software). While scientifically grounded, simulated sensor noise follows stationary Gaussian distributions, which do not fully replicate real-world transducer drift, vibration artifacts, or atmospheric particulate ingestion.
- **Single-Shift Tabular Nature of AI4I**: The UCI AI4I 2020 dataset consists of independent, single-record operational snapshots rather than longitudinal, multi-shift trajectories. Consequently, sequence models (LSTM) cannot be trained on AI4I without synthetic trajectory assumptions.

### 1.2 Cold-Start Sequence Horizon Requirements
- Deep sequence models require a minimum sequence history ($L=60$ cycles) to execute inference. For newly commissioned engines ($t < 60$ cycles), the framework must fall back to tabular XGBoost models or left-edge zero-padded tensors, introducing modest early-life uncertainty.

### 1.3 Stationary Operating Conditions (FD001 Focus)
- The core models in this project were trained on subset FD001 (single sea-level operating regime). Applying these exact checkpoints to multi-condition subsets (FD002/FD004 with 6 flight regimes) requires operational regime clustering or altitude-normalization preprocessing.

---

## 2. Promising Directions for Future Research

1. **Transformer & Mamba State-Space Architectures**:
   - Investigate Transformer self-attention (PatchTST, Informer) and selective state-space models (Mamba) to capture long-range temporal dependencies over $L > 200$ flight cycles with linear computational complexity.
2. **Transfer Learning Across Operational Regimes**:
   - Train domain-adversarial neural networks (DANN) to transfer degradation representations learned on sea-level engines (FD001) to multi-regime commercial fleets (FD002/FD004).
3. **Reinforcement Learning for Prescriptive Scheduling**:
   - Upgrade the rule-based maintenance recommendation engine into a Reinforcement Learning agent (PPO/DQN) optimizing economic utility functions (downtime cost vs. spare parts replacement cost).
4. **Physical Industrial IoT Ingestion (OPC-UA / MQTT)**:
   - Deploy native industrial bus connectors (OPC-UA, MQTT, Kafka) enabling live sensor ingestion directly from plant SCADA systems.
