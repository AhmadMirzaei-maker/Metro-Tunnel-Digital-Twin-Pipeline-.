# -*- coding: utf-8 -*-
"""
Synthetic Sensor Data Generator for Tehran Metro Tunnel
Calibrated with real-world vibration data.
Includes: Temperature, Z-axis Accelerometer, Extensometer.
Start Date: 2026-07-01 06:00:00
FIXED:
  1. track_ids and sensor_indices now generate 12 unique sensor combinations.
  2. Extensometer strain_peak formula corrected (was saturating at ±0.05).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

# ========================== Core Simulation Parameters ==========================

OPERATION_START_HOUR = 6
OPERATION_END_HOUR = 23
OPERATION_HOURS = OPERATION_END_HOUR - OPERATION_START_HOUR

SENSOR_SPACING_METERS = 30
TUNNEL_LENGTH_METERS = 150
NUM_SENSORS_PER_TRACK = int(TUNNEL_LENGTH_METERS / SENSOR_SPACING_METERS) + 1  # 6
NUM_TRACKS = 2
NUM_SENSORS_TOTAL = NUM_SENSORS_PER_TRACK * NUM_TRACKS  # 12

TRANSMISSION_INTERVAL_SECONDS = 10
TRAIN_SPEED_MPS = 10.0
SIMULATION_DAYS = 30

SAMPLES_PER_HOUR = 3600 / TRANSMISSION_INTERVAL_SECONDS
SAMPLES_PER_DAY = int(SAMPLES_PER_HOUR * OPERATION_HOURS)
TOTAL_SAMPLES = SAMPLES_PER_DAY * SIMULATION_DAYS

print("="*60)
print("Simulation Parameters (Calibrated - Z-axis only):")
print(f"   - Tunnel Length: {TUNNEL_LENGTH_METERS} m")
print(f"   - Sensor Spacing: {SENSOR_SPACING_METERS} m")
print(f"   - Sensors per Track: {NUM_SENSORS_PER_TRACK}")
print(f"   - Total Sensors: {NUM_SENSORS_TOTAL}")
print(f"   - Operation Hours: {OPERATION_START_HOUR}:00 to {OPERATION_END_HOUR}:00")
print(f"   - Transmission Interval: {TRANSMISSION_INTERVAL_SECONDS} s")
print(f"   - Train Speed: {TRAIN_SPEED_MPS} m/s")
print(f"   - Simulation Days: {SIMULATION_DAYS}")
print(f"   - Total Samples: {TOTAL_SAMPLES:,}")
print("="*60)

# ========================== Standard Thresholds ==========================
SENSOR_THRESHOLDS = {
    'temperature': {'min': 15.0, 'max': 45.0, 'warning_min': 18.0, 'warning_max': 42.0},
    'accelerometer_x': {'min': -1.5, 'max': 1.5, 'warning_min': -1.0, 'warning_max': 1.0},
    'accelerometer_y': {'min': -0.8, 'max': 0.8, 'warning_min': -0.5, 'warning_max': 0.5},
    'accelerometer_z': {'min': -2.0, 'max': 2.0, 'warning_min': -1.2, 'warning_max': 1.2},
    'extensometer': {'min': -0.05, 'max': 0.05, 'warning_min': -0.03, 'warning_max': 0.03}
}

# ========================== Calibration Function ==========================
def calibrate_from_real_data(excel_file, sheet_name='github.com-Metro_vibration_v1_z'):
    """
    Reads a real Excel file and extracts statistical parameters for simulation calibration.
    If the file does not exist, default parameters are used.
    """
    try:
        df_real = pd.read_excel(excel_file, sheet_name=sheet_name)
        if 'z_data/g' in df_real.columns:
            values = df_real['z_data/g'].dropna().values
            mean_val = np.mean(values)
            std_val = np.std(values)
            p95 = np.percentile(values, 95)
            p99 = np.percentile(values, 99)
            max_abs = np.max(np.abs(values))
            unique_vals = np.unique(np.round(values, 1))
            print(f"   - Calibration from {excel_file} performed.")
            print(f"     Mean: {mean_val:.3f}, Std Dev: {std_val:.3f}")
            print(f"     95th Percentile: {p95:.3f}, 99th Percentile: {p99:.3f}, Max Abs: {max_abs:.3f}")
            print(f"     Unique Values (Quantized): {unique_vals[:10]} ...")
            return {'mean': mean_val, 'std': std_val, 'p95': p95, 'p99': p99, 'max_abs': max_abs}
        else:
            print("   - Column 'z_data/g' not found in the Excel file. Using default values.")
            return None
    except Exception as e:
        print(f"   - Error reading Excel file: {e}. Using default values.")
        return None

# ========================== Data Generation Functions (Vectorized) ==========================

def generate_timestamps(start_date, num_samples, interval_sec):
    start = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    return [start + timedelta(seconds=i * interval_sec) for i in range(num_samples)]

def generate_all_sensor_data(num_samples, sensor_type, track_id, sensor_idx, seed=None, calibration_params=None):
    """
    Generates a complete array for a specific sensor type (Vectorized) with optional calibration.
    """
    if seed is not None:
        np.random.seed(seed + track_id * 100 + sensor_idx)
    
    position = sensor_idx * SENSOR_SPACING_METERS
    delay_samples = int(position / TRAIN_SPEED_MPS / TRANSMISSION_INTERVAL_SECONDS)
    time_indices = np.arange(num_samples)
    
    th = SENSOR_THRESHOLDS.get(sensor_type, {})
    
    # ===== Accelerometer Z (Vertical) with Calibration =====
    if sensor_type == "accelerometer_z":
        if calibration_params is None:
            impact_amplitude = 2.0
            impact_duration = 15
            noise_std = 0.02
            quant_step = 0.1
        else:
            impact_amplitude = min(calibration_params['max_abs'] * 1.2, 2.5)
            impact_duration = 15
            noise_std = calibration_params['std'] * 0.5 if calibration_params['std'] > 0.01 else 0.02
            quant_step = 0.1
        
        base = 0.0
        train_impact = np.zeros(num_samples)
        num_trains_per_day = np.random.randint(5, 9)
        for day in range(SIMULATION_DAYS):
            day_start = day * SAMPLES_PER_DAY
            train_times = np.random.randint(0, SAMPLES_PER_DAY, num_trains_per_day)
            for t0 in train_times:
                abs_start = day_start + t0
                amp = np.random.uniform(impact_amplitude * 0.5, impact_amplitude)
                if abs_start < num_samples:
                    for i in range(impact_duration):
                        idx = abs_start + i
                        if idx < num_samples:
                            pulse_val = amp * np.exp(-i / 5) * (1 + 0.3 * np.random.randn())
                            train_impact[idx] += pulse_val
        
        background_noise = np.random.normal(0, noise_std, num_samples)
        data = base + train_impact + background_noise
        data = np.clip(data, th['min'], th['max'])
        data = np.round(data / quant_step) * quant_step
        return data
    
    # ===== Temperature Data =====
    elif sensor_type == "temperature":
        normal_mean = (th.get('min', 0) + th.get('max', 0)) / 2
        normal_std = (th.get('max', 1) - th.get('min', 1)) / 6
        base_data = np.random.normal(normal_mean, normal_std, num_samples)
        base_data = np.clip(base_data, th.get('min', -1e6), th.get('max', 1e6))
        
        daily_cycle = 8 * np.sin(2 * np.pi * time_indices / SAMPLES_PER_DAY)
        seasonal_trend = 3 * np.sin(2 * np.pi * time_indices / (SAMPLES_PER_DAY * SIMULATION_DAYS))
        position_effect = -1.5 * np.sin(np.pi * position / TUNNEL_LENGTH_METERS)
        noise = np.random.normal(0, 0.2, num_samples)
        data = 25 + daily_cycle + seasonal_trend + position_effect + noise
        return np.clip(data, th['min'], th['max'])
    
    # ===== Extensometer Data (FIXED FORMULA) =====
    elif sensor_type == "extensometer":
        normal_mean = (th.get('min', 0) + th.get('max', 0)) / 2
        normal_std = (th.get('max', 1) - th.get('min', 1)) / 6
        base_data = np.random.normal(normal_mean, normal_std, num_samples)
        base_data = np.clip(base_data, th.get('min', -1e6), th.get('max', 1e6))
        
        # ===== FIXED: Gaussian pulse formula (was saturating at ±0.05) =====
        # The previous formula had a sign error causing exp() to blow up.
        # Now uses proper Gaussian: exp(-x² / (2σ²)) with σ = 400 samples
        phase = ((time_indices - delay_samples) % SAMPLES_PER_DAY) - SAMPLES_PER_DAY / 2
        strain_peak = 0.015 * np.exp(-(phase ** 2) / (2 * 400 ** 2))
        
        background = 0.002 * (1 + 0.3 * np.sin(np.pi * position / TUNNEL_LENGTH_METERS))
        trend = 0.0005 * np.sin(2 * np.pi * time_indices / (SAMPLES_PER_DAY * 15))
        noise = np.random.normal(0, 0.001, num_samples)
        data = strain_peak + background + trend + noise
        return np.clip(data, th['min'], th['max'])
    
    # ===== Other Sensors (Placeholder for future expansion) =====
    else:
        normal_mean = (th.get('min', 0) + th.get('max', 0)) / 2
        normal_std = (th.get('max', 1) - th.get('min', 1)) / 6
        base_data = np.random.normal(normal_mean, normal_std, num_samples)
        return np.clip(base_data, th.get('min', -1e6), th.get('max', 1e6))

def generate_maintenance_dates(start_date, num_samples):
    start = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    dates = []
    for i in range(num_samples):
        if i % 500 == 0 and i > 0:
            track = np.random.randint(0, NUM_TRACKS)
            sensor_idx = np.random.randint(0, NUM_SENSORS_PER_TRACK)
            day_offset = np.random.randint(0, SIMULATION_DAYS)
            hour_offset = np.random.randint(OPERATION_START_HOUR, OPERATION_END_HOUR)
            minute_offset = np.random.randint(0, 60)
            dt = start + timedelta(days=day_offset, hours=hour_offset, minutes=minute_offset)
            dates.append(f"{dt.strftime('%Y-%m-%d %H:%M:%S')}|Track{track+1}_Sensor{sensor_idx+1:02d}")
        else:
            dates.append("")
    return dates

# ========================== Data Generation Execution (Vectorized) ==========================

print("\nStarting synthetic data generation (Vectorized method)...")
start_date = "2026-07-01 06:00:00"

# Calibration from real file (if available)
calibration_params = None
real_file = "Metro_vibration_v1_z_axis_.xlsx"
if os.path.exists(real_file):
    print("\nReal data file found. Performing calibration...")
    calibration_params = calibrate_from_real_data(real_file)
else:
    print("\nReal data file not found. Using default values.")

# Create empty lists for columns
timestamps = generate_timestamps(start_date, TOTAL_SAMPLES, TRANSMISSION_INTERVAL_SECONDS)

# ===== FIX: generate 12 unique (track, sensor) combinations =====
track_ids = [(i // NUM_SENSORS_PER_TRACK) % NUM_TRACKS for i in range(TOTAL_SAMPLES)]
sensor_indices = [i % NUM_SENSORS_PER_TRACK for i in range(TOTAL_SAMPLES)]

# Generate Base DataFrame
df = pd.DataFrame({
    'Timestamp': timestamps,
    'Track': [f'Track_{t+1}' for t in track_ids],
    'Sensor_ID': [f'Sensor_{t+1:02d}_{s+1:02d}' for t, s in zip(track_ids, sensor_indices)],
    'Location_Meter': [s * SENSOR_SPACING_METERS for s in sensor_indices]
})

# --- Generate Temperature Data ---
print("   - Generating Temperature Data...")
temp_data = np.zeros(TOTAL_SAMPLES)
for t in range(NUM_TRACKS):
    for s in range(NUM_SENSORS_PER_TRACK):
        mask = (df['Track'] == f'Track_{t+1}') & (df['Sensor_ID'].str.endswith(f'_{s+1:02d}'))
        if mask.any():
            idx = mask[mask].index
            data = generate_all_sensor_data(len(idx), 'temperature', t, s, seed=42)
            temp_data[idx] = data
df['Sensor_Temperature'] = temp_data

# --- Generate Accelerometer Z Data (with Calibration) ---
print("   - Generating Accelerometer Z Data (Vertical) with Calibration...")
acc_z_data = np.zeros(TOTAL_SAMPLES)
for t in range(NUM_TRACKS):
    for s in range(NUM_SENSORS_PER_TRACK):
        mask = (df['Track'] == f'Track_{t+1}') & (df['Sensor_ID'].str.endswith(f'_{s+1:02d}'))
        if mask.any():
            idx = mask[mask].index
            data = generate_all_sensor_data(len(idx), 'accelerometer_z', t, s, seed=45, calibration_params=calibration_params)
            acc_z_data[idx] = data
df['Sensor_Accelerometer_Z'] = acc_z_data

# --- Generate Extensometer Data ---
print("   - Generating Extensometer Data...")
ext_data = np.zeros(TOTAL_SAMPLES)
for t in range(NUM_TRACKS):
    for s in range(NUM_SENSORS_PER_TRACK):
        mask = (df['Track'] == f'Track_{t+1}') & (df['Sensor_ID'].str.endswith(f'_{s+1:02d}'))
        if mask.any():
            idx = mask[mask].index
            data = generate_all_sensor_data(len(idx), 'extensometer', t, s, seed=46)
            ext_data[idx] = data
df['Sensor_Extensometer'] = ext_data

# --- Generate Maintenance Dates ---
print("   - Generating Maintenance Dates...")
df['Maintenance_Date'] = generate_maintenance_dates(start_date, TOTAL_SAMPLES)

# Apply Operational Hours Constraint
print("\nApplying operational hour constraint...")
df['Hour'] = df['Timestamp'].dt.hour + df['Timestamp'].dt.minute / 60.0
df = df[(df['Hour'] >= OPERATION_START_HOUR) & (df['Hour'] < OPERATION_END_HOUR)]
df = df.drop(columns=['Hour'])
df = df.reset_index(drop=True)
print(f"   - Final sample count: {len(df):,}")

# Save to Excel
print("\nSaving data to Excel file...")
output_file = "Sensor_Data.xlsx"

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='All_Sensors', index=False)
    
    # Sensor-specific sheets
    sensor_columns = {
        'Temperature': 'Sensor_Temperature',
        'Accelerometer_Z': 'Sensor_Accelerometer_Z',
        'Extensometer': 'Sensor_Extensometer'
    }
    for sheet_name, col_name in sensor_columns.items():
        temp_df = df[['Timestamp', 'Track', 'Sensor_ID', 'Location_Meter', col_name]].copy()
        temp_df.to_excel(writer, sheet_name=f'Sensor_{sheet_name}', index=False)
    
    # Thresholds sheet
    thresholds_df = pd.DataFrame([
        {'Sensor': k, 'Min': v['min'], 'Max': v['max'], 
         'Warning_Min': v['warning_min'], 'Warning_Max': v['warning_max']}
        for k, v in SENSOR_THRESHOLDS.items()
    ])
    thresholds_df.to_excel(writer, sheet_name='Standard_Thresholds', index=False)
    
    # Maintenance Records
    maintenance_df = df[df['Maintenance_Date'] != ''][['Timestamp', 'Track', 'Sensor_ID', 'Location_Meter', 'Maintenance_Date']].copy()
    if not maintenance_df.empty:
        maintenance_df.to_excel(writer, sheet_name='Maintenance_Records', index=False)
    
    # Simulation Parameters Summary
    summary = pd.DataFrame({
        'Parameter': ['Tunnel Length (m)', 'Sensor Spacing (m)', 'Sensors per Track', 
                      'Number of Tracks', 'Total Sensors', 'Operation Start Hour',
                      'Operation End Hour', 'Transmission Interval (s)', 'Train Speed (m/s)',
                      'Simulation Days', 'Total Samples', 'Start Date'],
        'Value': [TUNNEL_LENGTH_METERS, SENSOR_SPACING_METERS, NUM_SENSORS_PER_TRACK,
                  NUM_TRACKS, NUM_SENSORS_TOTAL, OPERATION_START_HOUR, OPERATION_END_HOUR,
                  TRANSMISSION_INTERVAL_SECONDS, TRAIN_SPEED_MPS, SIMULATION_DAYS,
                  len(df), start_date]
    })
    summary.to_excel(writer, sheet_name='Simulation_Parameters', index=False)

print(f"File successfully saved: {output_file}")
print(f"   - Records: {len(df):,}")
print(f"   - Path: {os.path.abspath(output_file)}")

print("\nSample of generated data (first 10 records):")
print(df.head(10).to_string())

print("\nData Statistics:")
print(f"   - Time Range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
print(f"   - Temperature Range: {df['Sensor_Temperature'].min():.2f} to {df['Sensor_Temperature'].max():.2f} °C")
print(f"   - Accelerometer Z Range: {df['Sensor_Accelerometer_Z'].min():.3f} to {df['Sensor_Accelerometer_Z'].max():.3f} g")
print(f"   - Extensometer Range: {df['Sensor_Extensometer'].min():.3f} to {df['Sensor_Extensometer'].max():.3f} mm")
print(f"   - Maintenance Records: {len(df[df['Maintenance_Date'] != ''])}")
print(f"   - Active Sensors: {df['Sensor_ID'].nunique()}")

print("\nData generation completed successfully.")