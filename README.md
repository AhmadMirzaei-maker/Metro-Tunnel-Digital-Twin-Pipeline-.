# Metro Tunnel Digital Twin Pipeline

This repository contains the Python source code for the paper:

**"Improved Predictive Maintenance Management of Metro Tunnels by Integrating Digital Twin, Process Mining, and Explainable AI"**  
*Journal of Information Technology in Construction (ITcon), 2026.*

## Contents

- `integrated_pipeline.py` — Main pipeline (Digital Twin + Process Mining + XAI)
- `Risk_Score.py` — Risk score calculation for each sensor
- `Sensor_Data.py` — Synthetic sensor data generation

## Requirements

- Python 3.9+
- pm4py
- xgboost
- shap
- lime
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- networkx

Install all dependencies with:

```bash
pip install -r requirements.txt
Generate synthetic sensor data:python Sensor_Data.py
Calculate risk scores:python Risk_Score.py
Run the integrated pipeline:python integrated_pipeline.py
Outputs
The pipeline generates:

Process mining models (DFG, Heuristics Net, Inductive Tree)

XGBoost risk predictions with SHAP/LIME explanations

Confusion matrix and performance metrics

BIM-ready CSV files with color-coded sensor health

Excel dashboards (EN and FA versions)

KPI comparison charts

Citation
If you use this code, please cite:

Mirzaei, A., Pourrostam, T., Jafari Nadooshan, M., Asghari, P., & Majrouhi Sardroud, J. (2026). Improved predictive maintenance management of metro tunnels by integrating digital twin, process mining, and explainable AI. Journal of Information Technology in Construction (ITcon), 31, 1-X. https://doi.org/10.36680/j.itcon.2026.0

License
The source code is licensed under the CC0-1.0 License.
The associated paper is published under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.
