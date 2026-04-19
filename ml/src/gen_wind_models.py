"""
gen_wind_models.py
──────────────────
Generates wind turbine ML models (sklearn RandomForest, equivalent quality to
the project's solar XGBoost models) and saves them as .pkl files.

Run from the ml/src directory:
    python gen_wind_models.py

Output files:
    wind_efficiency_model.pkl
    wind_feature_columns.pkl
    wind_suitability_model.pkl
    wind_site_feature_columns.pkl
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
import joblib

np.random.seed(42)
N = 5000

print("="*70)
print("WINDSENSE AI — Wind Turbine Model Training")
print("="*70)

# ── 1. Synthetic sensor dataset ──────────────────────────────────────────
wind_speed      = np.random.uniform(2, 25, N)
wind_direction  = np.random.uniform(0, 360, N)
turbine_rpm     = wind_speed * 4 + np.random.normal(0, 2, N)
blade_pitch     = np.clip(np.random.normal(5, 3, N), 0, 90)
gearbox_temp    = 60 + wind_speed * 1.2 + np.random.normal(0, 5, N)
generator_temp  = 70 + wind_speed * 1.5 + np.random.normal(0, 6, N)
vibration_level = np.random.uniform(0.1, 2.0, N)
bearing_temp    = 45 + wind_speed * 0.8 + np.random.normal(0, 4, N)
ambient_temp    = np.random.normal(25, 10, N)
air_density     = 1.225 - (ambient_temp - 15) * 0.004
humidity        = np.random.uniform(20, 95, N)

# Efficiency (Betz-limited, degraded by heat/vibration)
real_efficiency = np.clip(
    0.42 - 0.01*(vibration_level-1) - 0.005*np.maximum(gearbox_temp-80,0)
    - 0.008*np.maximum(generator_temp-90,0) - 0.003*np.maximum(bearing_temp-70,0)
    + np.random.normal(0, 0.02, N),
    0.10, 0.50
)
real_efficiency = np.where(vibration_level>1.5, real_efficiency-0.05, real_efficiency)
real_efficiency = np.where(gearbox_temp>85,     real_efficiency-0.04, real_efficiency)
real_efficiency = np.where(generator_temp>95,   real_efficiency-0.06, real_efficiency)
real_efficiency = np.where(bearing_temp>75,     real_efficiency-0.03, real_efficiency)
real_efficiency = np.clip(real_efficiency, 0.05, 0.52)

# Site suitability
slope        = np.random.uniform(0, 45, N)
elevation    = np.random.uniform(50, 3000, N)
turb_intens  = vibration_level / np.maximum(wind_speed, 0.1) * 100
yearly_wind  = wind_speed + np.random.normal(0, 1, N)
site_label   = ((yearly_wind>6)&(turb_intens<20)&(slope<30)&(elevation<2500)).astype(int)

df = pd.DataFrame({
    'wind_speed': wind_speed, 'wind_direction': wind_direction,
    'turbine_rpm': turbine_rpm, 'blade_pitch': blade_pitch,
    'gearbox_temp': gearbox_temp, 'generator_temp': generator_temp,
    'vibration_level': vibration_level, 'bearing_temp': bearing_temp,
    'ambient_temp': ambient_temp, 'air_density': air_density,
    'humidity': humidity, 'slope': slope, 'elevation': elevation,
    'turbulence_intensity': turb_intens, 'yearly_wind_speed': yearly_wind,
    'efficiency': real_efficiency, 'site_label': site_label,
}).fillna(0)

print(f"Dataset: {df.shape[0]} rows  |  Efficiency {real_efficiency.min():.3f}–{real_efficiency.max():.3f}  |  {site_label.mean()*100:.1f}% suitable sites")

# ── 2. Efficiency model ──────────────────────────────────────────────────
EFF_FEATURES = ['wind_speed','wind_direction','turbine_rpm','blade_pitch',
                'gearbox_temp','generator_temp','vibration_level','bearing_temp',
                'ambient_temp','air_density','humidity']

X_eff = df[EFF_FEATURES]; y_eff = df['efficiency']
Xtr,Xte,ytr,yte = train_test_split(X_eff, y_eff, test_size=0.2, random_state=42)
model_eff = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
model_eff.fit(Xtr, ytr)
yp = model_eff.predict(Xte)
print(f"\nEfficiency Model  →  R²: {r2_score(yte,yp):.4f}  |  MAE: {mean_absolute_error(yte,yp):.4f}")

# ── 3. Site suitability model ─────────────────────────────────────────────
SITE_FEATURES = ['wind_speed','wind_direction','turbulence_intensity',
                 'slope','elevation','humidity','ambient_temp',
                 'air_density','yearly_wind_speed']

X_site = df[SITE_FEATURES]; y_site = df['site_label']
Xtr2,Xte2,ytr2,yte2 = train_test_split(X_site, y_site, test_size=0.2, random_state=42)
model_site = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
model_site.fit(Xtr2, ytr2)
acc = accuracy_score(yte2, model_site.predict(Xte2))
print(f"Site Suitability  →  Accuracy: {acc:.4f}")

# ── 4. Save ───────────────────────────────────────────────────────────────
joblib.dump(model_eff,    "wind_efficiency_model.pkl")
joblib.dump(EFF_FEATURES, "wind_feature_columns.pkl")
joblib.dump(model_site,   "wind_suitability_model.pkl")
joblib.dump(SITE_FEATURES,"wind_site_feature_columns.pkl")

print("\n" + "="*70)
print("SUCCESS — 4 wind turbine .pkl files saved:")
print("  wind_efficiency_model.pkl")
print("  wind_feature_columns.pkl")
print("  wind_suitability_model.pkl")
print("  wind_site_feature_columns.pkl")
print("="*70)
