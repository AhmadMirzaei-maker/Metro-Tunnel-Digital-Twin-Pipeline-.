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
