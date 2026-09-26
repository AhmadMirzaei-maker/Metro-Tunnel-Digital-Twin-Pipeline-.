# -*- coding: utf-8 -*-
"""
Risk Score Calculator for Tunnel Sensors
Calculates Risk_Score based on the generated Sensor_Data.xlsx
FIXED:
  - Robust header auto-detection for Sensor_Final_Schedule.xlsx
    (works whether or not the file has a title row).
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')


# ============================ Robust Schedule Reader ============================
def read_schedule_robust(path, sheet_name='Sensor_Final_Schedule'):
    """
    Reads a schedule Excel file regardless of whether the first row is a title
    or the actual header. Finds the row containing 'Sensor_ID' and uses it as header.
    """
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    header_row_idx = None
    for i in range(min(5, len(raw))):
        row_values = [str(v).strip() for v in raw.iloc[i].tolist()]
        if 'Sensor_ID' in row_values:
            header_row_idx = i
            break
    if header_row_idx is None:
        raise ValueError(
            f"Could not find a 'Sensor_ID' column in {path} (sheet: {sheet_name})."
        )
    raw.columns = [str(v).strip() for v in raw.iloc[header_row_idx].tolist()]
    df = raw.iloc[header_row_idx + 1:].reset_index(drop=True)
    df = df.dropna(how='all').reset_index(drop=True)
    return df


# ============================ Load Data ============================
print("Loading data...")

SENSOR_DATA_FILE = "Sensor_Data.xlsx"
if not os.path.exists(SENSOR_DATA_FILE):
    raise FileNotFoundError(f"File {SENSOR_DATA_FILE} not found. Please generate the data first.")

df_sensors = pd.read_excel(SENSOR_DATA_FILE, sheet_name='All_Sensors')
print(f"Sensor data loaded: {len(df_sensors):,} records")

# ============================ Load or Generate Schedule ============================
SCHEDULE_FILE = "Sensor_Final_Schedule.xlsx"
if os.path.exists(SCHEDULE_FILE):
    df_schedule = read_schedule_robust(SCHEDULE_FILE, 'Sensor_Final_Schedule')
    print(f"Schedule file loaded: {len(df_schedule)} sensors")
    if 'Risk_Score' not in df_schedule.columns:
        df_schedule['Risk_Score'] = np.nan
else:
    print("Schedule file not found. Generating schedule with 36 sensors...")
    sensors = []
    sensor_types = {'A': 'Accelerometer', 'E': 'Extensometer', 'T': 'Temperature'}
    for type_code, type_name in sensor_types.items():
        for track in [1, 2]:
            for idx in range(1, 7):
                sensor_id = f"{type_code}-{track:02d}-{idx:02d}"
                location = (idx - 1) * 30
                sensors.append({
                    "Sensor_ID": sensor_id,
                    "Sensor_Type": type_name,
                    "Track_Number": track,
                    "Location_Meter": location,
                    "Sensor_Status": "Active",
                    "Installation_Date": "2026-07-01 00:00:00"
                })
    df_schedule = pd.DataFrame(sensors)
    print(f"Schedule generated with {len(df_schedule)} sensors")
    df_schedule['Risk_Score'] = np.nan

# ============================ Matching Sensor IDs ============================
print("\nMatching sensor IDs...")

df_sensors['Track_Num'] = df_sensors['Track'].str.extract(r'(\d+)').astype(int)
df_sensors['Sensor_Index'] = df_sensors['Sensor_ID'].str.extract(r'_(\d+)_(\d+)')[1].astype(int)
df_sensors['Sensor_Key'] = df_sensors.apply(
    lambda r: f"{r['Track_Num']:02d}_{r['Sensor_Index']:02d}", axis=1
)

if 'Sensor_ID' not in df_schedule.columns:
    raise KeyError("Column 'Sensor_ID' not found in the Schedule file.")

df_schedule['Track_Num'] = df_schedule['Sensor_ID'].str.split('-').str[1].astype(int)
df_schedule['Sensor_Index'] = df_schedule['Sensor_ID'].str.split('-').str[2].astype(int)
df_schedule['Sensor_Key'] = df_schedule.apply(
    lambda r: f"{r['Track_Num']:02d}_{r['Sensor_Index']:02d}", axis=1
)
df_schedule['Sensor_Type_Short'] = df_schedule['Sensor_ID'].str.split('-').str[0]

# ============================ Calculate Risk_Score ============================
print("\nCalculating Risk_Score for each sensor...")

THRESHOLDS = {
    'T': {'col': 'Sensor_Temperature', 'warn_min': 18, 'warn_max': 42},
    'A': {'col': 'Sensor_Accelerometer_Z', 'warn_min': -1.2, 'warn_max': 1.2},
    'E': {'col': 'Sensor_Extensometer', 'warn_min': -0.03, 'warn_max': 0.03}
}

risk_scores = []

for _, row_sched in df_schedule.iterrows():
    sensor_key = row_sched['Sensor_Key']
    sensor_type = row_sched['Sensor_Type_Short']

    sensor_data = df_sensors[df_sensors['Sensor_Key'] == sensor_key]
    if sensor_data.empty:
        risk_scores.append(0.0)
        continue

    col_info = THRESHOLDS.get(sensor_type)
    if col_info is None:
        risk_scores.append(0.0)
        continue

    col_name = col_info['col']
    warn_min = col_info['warn_min']
    warn_max = col_info['warn_max']

    values = sensor_data[col_name].dropna().values
    if len(values) == 0:
        risk_scores.append(0.0)
        continue

    mean_val = np.mean(values)
    std_val = np.std(values)
    max_abs = np.max(np.abs(values))
    p95 = np.percentile(np.abs(values), 95)
    out_of_warn = np.sum((values < warn_min) | (values > warn_max)) / len(values)

    if sensor_type == 'T':
        max_abs_ref, std_ref, p95_ref = 50.0, 10.0, 40.0
    elif sensor_type == 'A':
        max_abs_ref, std_ref, p95_ref = 2.5, 0.5, 1.8
    else:
        max_abs_ref, std_ref, p95_ref = 0.06, 0.01, 0.045

    norm_max_abs = min(max_abs / max_abs_ref, 1.0)
    norm_std = min(std_val / std_ref, 1.0)
    norm_p95 = min(p95 / p95_ref, 1.0)
    norm_out = min(out_of_warn * 5, 1.0)

    risk_score = (0.3 * norm_max_abs + 0.2 * norm_std +
                  0.2 * norm_p95 + 0.3 * norm_out)

    risk_score = np.clip(risk_score, 0.0, 1.0)
    risk_scores.append(round(risk_score, 4))

df_schedule['Risk_Score'] = risk_scores
df_schedule['Risk_Score'] = df_schedule['Risk_Score'].replace(0, 0.05)

if 'Line' in df_schedule.columns:
    df_schedule.drop(columns=['Line'], inplace=True)

# ============================ Save ============================
print("\nSaving Schedule file with Risk_Score column...")

output_file = "Sensor_Final_Schedule_with_Risk.xlsx"
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    df_schedule.to_excel(writer, sheet_name='Sensor_Final_Schedule', index=False)

print(f"File successfully saved: {output_file}")
print(f"   - Total sensors: {len(df_schedule)}")
print(f"   - Path: {os.path.abspath(output_file)}")

print("\nSample (first 10 sensors):")
print(df_schedule[['Sensor_ID', 'Sensor_Type', 'Track_Number', 'Location_Meter', 'Risk_Score']].head(10).to_string())

print("\nRisk_Score Statistics:")
print(f"   - Minimum: {df_schedule['Risk_Score'].min():.4f}")
print(f"   - Maximum: {df_schedule['Risk_Score'].max():.4f}")
print(f"   - Mean: {df_schedule['Risk_Score'].mean():.4f}")
print(f"   - Std Dev: {df_schedule['Risk_Score'].std():.4f}")

print("\nProcess completed successfully.")