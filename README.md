# Metro Tunnel Digital Twin Pipeline

[![DOI](https://img.shields.io/badge/DOI-10.17632%2Fttc9ycvrx8.1-blue)](https://doi.org/10.17632/ttc9ycvrx8.1)

This repository contains the Python source code for the paper:

**"Improved Predictive Maintenance Management of Metro Tunnels by Integrating Digital Twin, Process Mining, and Explainable AI"**  
*Journal of Information Technology in Construction (ITcon), 2026.*

---

## Data Availability

The dataset and source code associated with this work are available at:

- **Mendeley Data (DOI):** https://doi.org/10.17632/ttc9ycvrx8.1
- **GitHub Repository:** https://github.com/AhmadMirzaei-maker/Metro-Tunnel-Digital-Twin-Pipeline-

---

## Contents

- `integrated_pipeline.py` — Main pipeline (Digital Twin + Process Mining + XAI)
- `Risk_Score.py` — Risk score calculation for each sensor
- `Sensor_Data.py` — Synthetic sensor data generation
- `requirements.txt` — Python dependencies
- `README.md` — This file
- `LICENSE` — CC0-1.0 License

---

## Requirements

- Python 3.9+
- All dependencies are listed in `requirements.txt`

Install all dependencies with:

```bash
pip install -r requirements.txt
Reproduction Steps
Generate synthetic sensor data:

bash
python Sensor_Data.py
Calculate risk scores:

bash
python Risk_Score.py
Run the integrated pipeline:

bash
python integrated_pipeline.py
Outputs
The pipeline generates the following outputs in the outputs/ folder:

Process Mining
dfg_networkx_FINAL.png — Directly-Follows Graph

heuristics_net.png — Heuristics Net

inductive_tree.png — Inductive Process Tree

event_log.csv — Full event log (936,292 events)

Machine Learning and XAI
pred_vs_actual_Training.png — Predicted vs. actual (training)

pred_vs_actual_Test.png — Predicted vs. actual (test)

shap_summary_top10.png — SHAP feature importance

confusion_matrix.png — Event-level confusion matrix

Risk Analysis
risk_level_pie_EN.png — Sensor risk distribution

spatial_risk_heatmap.png — Spatial risk distribution

validation_comparison.png — KPI comparison chart

Deliverables
Digital_Twin_Complete_EN.xlsx — Excel dashboard (English)

Digital_Twin_Complete_FA.xlsx — Excel dashboard (Persian)

bim_ready_sensors.csv — BIM-ready CSV with color coding

Citation
If you use this code or data in your research, please cite:

Mirzaei, A., Pourrostam, T., Jafari Nadooshan, M., Asghari, P., & Majrouhi Sardroud, J. (2026). Improved predictive maintenance management of metro tunnels by integrating digital twin, process mining, and explainable AI. Journal of Information Technology in Construction (ITcon), 31, 1-X. https://doi.org/10.36680/j.itcon.2026.0

License
The source code in this repository is licensed under the CC0-1.0 License.

The associated paper is published under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.
