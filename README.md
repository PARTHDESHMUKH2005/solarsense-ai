<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1e3c72,50:2a5298,100:ffc107&height=220&section=header&text=SolarSense%20AI&fontSize=72&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Predict.%20Explain.%20Prevent.&descAlignY=60&descSize=22&descColor=ffd166" width="100%"/>

<br/>

<img src="https://img.icons8.com/fluency/96/sun-energy.png" alt="SolarSense AI" width="96"/>

<br/><br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-Backend-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML%20Engine-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-7C4DFF?style=for-the-badge&logo=python&logoColor=white)](https://shap.readthedocs.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

<br/>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Thapar University](https://img.shields.io/badge/Thapar%20Institute-Engineering%20%26%20Technology-blue?style=for-the-badge)](https://thapar.edu)
[![Team: GreenMind AI](https://img.shields.io/badge/Team-GreenMind%20AI-2ea44f?style=for-the-badge&logo=github)](https://github.com/PARTHDESHMUKH2005/solarsense-ai)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen.svg?style=for-the-badge)](http://makeapullrequest.com)

<br/>

> ### *"By combining artificial intelligence with renewable infrastructure, our solution enables smarter operations, higher sustainability, and a cleaner energy future."*
> — **GreenMind AI Team**

<br/>

[🚀 Quickstart](#-quick-start) · [🏗️ Architecture](#%EF%B8%8F-system-architecture) · [📂 Codebase Walkthrough](#-codebase-walkthrough) · [📊 ML Models](#-ml-models--training) · [🔮 Future Roadmap](#-roadmap)

</div>

---

## 📋 Table of Contents

- [🌍 Problem Statement](#-problem-statement)
- [💡 Our Solution](#-our-solution)
- [✨ Key Features](#-key-features)
- [🛠️ Tech Stack](#%EF%B8%8F-tech-stack)
- [🏗️ System Architecture](#%EF%B8%8F-system-architecture)
- [📂 Codebase Walkthrough](#-codebase-walkthrough)
- [📊 ML Models & Training](#-ml-models--training)
- [🗃️ Datasets](#%EF%B8%8F-datasets)
- [🔍 Explainability with SHAP](#-explainability-with-shap)
- [🎨 Frontend Dashboard](#-frontend-dashboard)
- [🚀 Quick Start](#-quick-start)
- [📈 Performance Metrics](#-performance-metrics)
- [🔮 Roadmap](#-roadmap)
- [👥 Team](#-team)

---

## 🌍 Problem Statement

> **Predictive Maintenance for Renewable Infrastructure** — Create a model that predicts failures or efficiency drops in solar panels, wind turbines, or batteries using sensor data and weather patterns.

Solar energy infrastructure is at the heart of the global clean energy transition. Yet across large-scale deployments — from desert megafarms to rooftop urban grids — a critical gap remains: **operators react to failures instead of preventing them.**

### 🏜️ Renewable Energy Challenges in Extreme Climates (Dubai Context)

<table>
<tr>
<td width="50%" valign="top">

**🌡️ Extreme Heat Accelerates Degradation**
- Dubai summers reach **45–50°C**, pushing panels beyond thermal limits
- High temperatures cause faster panel aging, inverter failures, sensor malfunctions, and power losses due to overheating
- Panel surface temperatures can exceed **70°C**, causing efficiency losses of 10–25%

**🌪️ Dust & Sandstorm Soiling Losses**
- Frequent desert sandstorms cause rapid soiling accumulation on panel surfaces
- Operators lack intelligent systems to predict *when* cleaning is actually needed vs. scheduled blindly
- Soiling alone can reduce output by **5–10% per week** in high-dust environments

</td>
<td width="50%" valign="top">

**🎯 Sustainability & Net-Zero Targets at Risk**
- UAE's **Net Zero 2050** roadmap depends on reliable, high-output solar installations
- Performance losses reduce carbon reduction impact, ROI on renewable investments, and public trust in green infrastructure

**🤖 Lack of Integrated AI-Based Monitoring**
- Existing tools display raw data without intelligence
- Engineers manually analyze hundreds of parameters across thousands of panels
- This leads to **delayed decisions**, missed faults, and expensive reactive maintenance

</td>
</tr>
</table>

```
THE COST OF REACTIVE MAINTENANCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ❌  Failures detected AFTER they occur
  ❌  Revenue lost during unplanned downtime
  ❌  Guesswork-driven cleaning & inspection schedules
  ❌  No early warning system for component degradation
  ❌  Manual analysis of complex multi-sensor datasets
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✅  What if we could predict failures BEFORE they happen?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 💡 Our Solution

**SolarSense AI** is an end-to-end AI-driven predictive maintenance platform built by **Team GreenMind AI** at Thapar Institute of Engineering and Technology. It ingests real-time sensor readings and weather data to forecast efficiency drops and equipment failures — days before they cause costly downtime.

<div align="center">

| 🔮 Predict | 🔍 Explain | 🤖 Recommend | 📊 Visualize |
|:---:|:---:|:---:|:---:|
| Forecast efficiency drops & failure risk using XGBoost ML models | Understand *why* performance degrades using SHAP explainability | Rule-based and AI-powered maintenance action suggestions | Interactive dashboard with 3D solar panel visualization, charts & maps |

</div>

The MVP focuses on **solar panels** and demonstrates predictions, risk scoring, site suitability analysis, and AI-powered insights through a clean web dashboard — all backed by a Flask REST API and trained XGBoost models.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔮 **Efficiency Prediction** | XGBoost regressor predicts panel efficiency (0–1 scale) from sensor + weather inputs |
| ⚠️ **Risk Scoring** | Computed risk score (0–100%) derived from predicted efficiency with failure flagging at <75% |
| 🗺️ **Site Suitability Analysis** | XGBoost classifier answers: *"Is this location suitable for solar installation?"* using quarterly climate data |
| 🧠 **SHAP Explainability** | Tree-based SHAP values reveal exactly which sensor/weather features are driving efficiency up or down |
| 💬 **AI Insights Engine** | Rule-based insight generator (with optional GPT-4 integration) provides plain-language maintenance recommendations |
| 📐 **3D Solar Panel Visualization** | Real-time Three.js 3D model that responds to efficiency values — tilt, glow, and color change with predictions |
| 🗺️ **Geospatial Mapping** | Leaflet.js interactive map for site location analysis and visualization |
| 📈 **Cost Analytics** | Monthly and cumulative maintenance expenditure charts using Chart.js |
| 🔌 **REST API** | Clean Flask API with `/predict`, `/health`, and static file serving endpoints |
| ⚡ **No New Hardware Required** | Works entirely with existing sensor data infrastructure — pure software solution |

---

## 🛠️ Tech Stack

<div align="center">

### Backend & Machine Learning

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge)](https://xgboost.readthedocs.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![SHAP](https://img.shields.io/badge/SHAP-7C4DFF?style=for-the-badge)](https://shap.readthedocs.io)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![joblib](https://img.shields.io/badge/joblib-Model%20Persistence-blueviolet?style=for-the-badge)](https://joblib.readthedocs.io)

### Frontend & Visualization

[![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/HTML)
[![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/CSS)
[![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
[![Bootstrap](https://img.shields.io/badge/Bootstrap%205-7952B3?style=for-the-badge&logo=bootstrap&logoColor=white)](https://getbootstrap.com)
[![Three.js](https://img.shields.io/badge/Three.js-3D%20Visualization-black?style=for-the-badge&logo=three.js&logoColor=white)](https://threejs.org)
[![Chart.js](https://img.shields.io/badge/Chart.js-FF6384?style=for-the-badge&logo=chart.js&logoColor=white)](https://chartjs.org)
[![Leaflet](https://img.shields.io/badge/Leaflet-199900?style=for-the-badge&logo=leaflet&logoColor=white)](https://leafletjs.com)

### AI / GenAI (Optional)

[![OpenAI](https://img.shields.io/badge/OpenAI%20GPT--4-Optional%20Integration-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com)
[![LangChain](https://img.shields.io/badge/LangChain-AI%20Orchestration-1C3C3C?style=for-the-badge)](https://langchain.com)

### Development & Data

[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![MySQL](https://img.shields.io/badge/MySQL-4479A1?style=for-the-badge&logo=mysql&logoColor=white)](https://mysql.com)

</div>

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SOLARSENSE AI — SYSTEM OVERVIEW                  │
└─────────────────────────────────────────────────────────────────────────┘

  ┌────────────────────┐     ┌────────────────────┐     ┌──────────────────┐
  │   🌤️  DATA INPUTS  │     │  ⚙️  DATA PIPELINE  │     │ 🤖  ML MODELS   │
  │                    │     │                    │     │                  │
  │  Sensor Data:      │────▶│  Feature           │────▶│  XGBoost         │
  │  • Voltage         │     │  Engineering       │     │  Regressor       │
  │  • Current         │     │                    │     │  (Efficiency)    │
  │  • Panel Temp      │     │  Normalization &   │     │                  │
  │  • Dust Index      │     │  Scaling           │     │  XGBoost         │
  │                    │     │                    │     │  Classifier      │
  │  Weather Data:     │     │  Outlier &         │     │  (Suitability)   │
  │  • Temperature     │     │  Missing Value     │     │                  │
  │  • Humidity        │     │  Handling          │     │  SHAP Tree       │
  │  • Cloud Cover     │     │                    │     │  Explainer       │
  │  • Precipitation   │     │  Quarterly         │     │                  │
  │  • Wind Speed      │     │  Aggregation       │     │                  │
  │                    │     │                    │     │                  │
  │  Site Data:        │     │  Panel Temp        │     │                  │
  │  • GHI, DNI, DHI   │     │  Derivation:       │     │                  │
  │  • Location        │     │  T + (Irr/800)×20  │     │                  │
  │  • Capacity        │     │                    │     │                  │
  └────────────────────┘     └────────────────────┘     └──────────────────┘
             │                                                    │
             └─────────────────────────────────────────────────▶ │
                                                                  ▼
  ┌────────────────────┐     ┌────────────────────┐     ┌──────────────────┐
  │  📱 DASHBOARD UI   │     │  🔌 FLASK REST API  │     │  🧠 INSIGHTS    │
  │                    │     │                    │     │                  │
  │  3D Solar Panel    │◀────│  POST /predict     │◀────│  SHAP Values     │
  │  (Three.js)        │     │  GET  /health      │     │  → Feature       │
  │                    │     │  GET  /            │     │    Importance    │
  │  Efficiency Gauge  │     │                    │     │                  │
  │  Risk Score        │     │  CORS enabled      │     │  Rule-Based      │
  │  Failure Flag      │     │  JSON I/O          │     │  Insights        │
  │                    │     │  Input validation  │     │  Engine          │
  │  SHAP Bar Chart    │     │  Error handling    │     │                  │
  │  Maintenance       │     │                    │     │  (Optional)      │
  │  Cost Charts       │     │                    │     │  GPT-4 via       │
  │                    │     │                    │     │  OpenAI API      │
  │  Leaflet Map       │     │                    │     │                  │
  │  Site Suitability  │     │                    │     │                  │
  └────────────────────┘     └────────────────────┘     └──────────────────┘
```

### 🔄 Request / Prediction Lifecycle

```
User enters sensor inputs in dashboard
            │
            ▼
    JavaScript collects form values
    (temperature, humidity, irradiance,
     dust_index, cloudcover, precip,
     wind_speed, voltage, current)
            │
            ▼ HTTP POST /predict (JSON)
    Flask app.py receives request
            │
            ├──▶ Validates required fields
            ├──▶ Derives panel_temp = T + (irradiance/800) × 20
            ├──▶ Fills missing features with defaults
            ├──▶ Reindexes DataFrame to match feature_columns.pkl
            │
            ├──▶ MODEL 1: efficiency_model.pkl
            │        XGBoost Regressor → predicted_efficiency (0.0–1.0)
            │        risk_score = (1 - efficiency) × 100
            │        failure_flag = efficiency < 0.75
            │
            ├──▶ SHAP: explain.py
            │        TreeExplainer → per-feature SHAP values
            │        (fallback to rule-based if SHAP unavailable)
            │
            ├──▶ INSIGHTS: get_genai_insights()
            │        Rule-based analysis of top SHAP factors
            │        (or GPT-4 via OpenAI if USE_OPENAI=True)
            │
            └──▶ MODEL 2: suitability_model.pkl
                     XGBoost Classifier → Site suitable? (Yes/No)
                     Uses quarterly climate features
            │
            ▼ JSON Response
    {
      predicted_efficiency,
      risk_score,
      failure_flag,
      explanation (SHAP dict),
      insights_and_suggestions,
      recommended_action,
      suitability
    }
            │
            ▼
    Dashboard updates:
    • Efficiency display + color coding
    • Risk score badge
    • SHAP bar chart
    • Action recommendation
    • 3D panel visual adapts
    • Site suitability shown
```

---

## 📂 Codebase Walkthrough

```
solarsense-ai/
│
├── 📄 main.py                          # Streamlit prototype (Gemini-based demo app)
├── 📄 requirements.txt                 # Full Python dependency list
├── 📄 README.md                        # This file
│
├── 📁 data/
│   ├── 📁 raw/
│   │   ├── Solar_Sites_Dataset_India.csv     # Real solar site suitability data (India)
│   │   ├── Wind_Sites_Dataset_India.csv      # Wind site data
│   │   └── indian_weather_data.csv           # Historical Indian weather records
│   ├── 📁 processed/
│   │   └── final_dataset.csv                 # Merged + cleaned training dataset
│   └── 📁 synthetic/
│       └── sensor_data.csv                   # 5,000-row synthetic sensor dataset
│
└── 📁 ml/
    ├── 📄 combined.py                        # Data combination/merging script
    │
    ├── 📁 notebooks/
    │   └── 📓 eda.ipynb                      # Exploratory Data Analysis notebook
    │
    └── 📁 src/                               # 🔑 Core application source
        ├── 📄 app.py                         # Flask REST API server (main entry point)
        ├── 📄 train_ml.py                    # Model training script (XGBoost)
        ├── 📄 explain.py                     # SHAP explainability module
        ├── 📄 f_e.py                         # Feature engineering utilities
        ├── 📄 generate_syndata.py            # Synthetic sensor data generator
        ├── 📄 procfile                       # Deployment process definition
        ├── 📄 requirements.txt               # ML-specific dependencies
        ├── 📄 solar_panel_combined_dataset.csv  # Combined training dataset
        │
        ├── 📦 efficiency_model.pkl           # Trained XGBoost efficiency regressor
        ├── 📦 suitability_model.pkl          # Trained XGBoost site classifier
        ├── 📦 feature_columns.pkl            # Saved feature column order (efficiency)
        ├── 📦 site_feature_columns.pkl       # Saved feature column order (suitability)
        │
        └── 📁 static/
            ├── 🌐 main.html                  # Full frontend dashboard (Bootstrap + Three.js)
            └── 📄 solar_3d.js               # Three.js 3D solar panel animation module
```

### File-by-File Deep Dive

#### `ml/src/app.py` — Flask REST API

The heart of the backend. Loads both trained models at startup, exposes three routes, and orchestrates the full prediction pipeline:

```python
# Model loading at startup
model_eff    = joblib.load("efficiency_model.pkl")      # XGBoost Regressor
model_site   = joblib.load("suitability_model.pkl")     # XGBoost Classifier
feature_cols = joblib.load("feature_columns.pkl")       # Feature alignment

# Core prediction logic in POST /predict:
data['panel_temp'] = data['temperature'] + (data['irradiance'] / 800.0) * 20
efficiency  = float(model_eff.predict(df)[0])           # 0.0 → 1.0
risk_score  = round((1 - efficiency) * 100, 2)          # 0 → 100%
failure_flag = efficiency < 0.75                         # Boolean alert
explanation = explain_prediction(data)                  # SHAP values dict
insights    = get_genai_insights(explanation, efficiency) # AI text
suitability = model_site.predict(df_site)[0]            # 'Yes' / 'No'
```

**Risk Action Thresholds:**

| Risk Score | Recommended Action |
|---|---|
| `< 30%` | ✅ Monitor closely — System performing well |
| `30–60%` | ⚠️ Optimize — Consider maintenance and cleaning |
| `> 60%` | 🚨 Immediate action required — Failure risk detected! |

---

#### `ml/src/train_ml.py` — Model Training

Trains both ML models from the combined solar dataset using XGBoost:

```python
# MODEL 1: Efficiency Regressor
xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=5,
    objective='reg:squarederror'
)
# Evaluated with: R² score (target > 0.94), MAE

# MODEL 2: Site Suitability Classifier
xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4
)
# Evaluated with: Accuracy score
```

**Training pipeline:**
1. Load `solar_panel_combined_dataset.csv`
2. Map `Label (Yes/No)` → binary (1/0)
3. Drop ID columns, apply one-hot encoding to categoricals
4. Fill remaining NaN values with 0
5. Split 80/20 train/test
6. Fit models, evaluate, serialize to `.pkl` via `joblib`

---

#### `ml/src/explain.py` — SHAP Explainability

Uses **SHAP TreeExplainer** to compute Shapley values — a game-theoretic approach to explaining model predictions:

```python
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(df)
# Returns: {feature_name: shap_value, ...}
# Positive SHAP value → feature increases efficiency
# Negative SHAP value → feature decreases efficiency
```

Falls back gracefully to a rule-based explanation engine when SHAP is unavailable (e.g., in resource-constrained environments).

---

#### `ml/src/generate_syndata.py` — Synthetic Data Generator

Generates 5,000 rows of realistic sensor data with injected degradation patterns:

```python
# Realistic degradation simulation
df.loc[df["dust_index"] > 0.7, "efficiency"] -= 0.15   # Dust soiling
df.loc[df["panel_temp"] > 55, "efficiency"] -= 0.10    # Thermal stress
```

---

#### `ml/src/static/main.html` — Frontend Dashboard

A fully self-contained single-page application with glassmorphism design, animated 3D panel, and live API calls:

- **Bootstrap 5** for responsive layout
- **Three.js (r128)** for real-time 3D solar panel model
- **Chart.js** for efficiency, cost, and maintenance charts
- **Leaflet.js** for site geospatial map
- **Font Awesome 6** for iconography
- Custom CSS with `backdrop-filter` glassmorphism effects

---

## 📊 ML Models & Training

### Model 1 — Efficiency Predictor (XGBoost Regressor)

**Goal:** Predict solar panel efficiency as a continuous value between 0 and 1.

**Input Features:**

| Feature | Type | Description |
|---|---|---|
| `temperature` | float | Ambient air temperature (°C) |
| `humidity` | float | Relative humidity (%) |
| `irradiance` | float | Solar irradiance (W/m²) |
| `dust_index` | float | Surface soiling index (0–1) |
| `panel_temp` | float | **Derived**: `T + (irr/800) × 20` |
| `cloudcover` | float | Cloud cover fraction |
| `precip` | float | Precipitation (mm) |
| `wind_speed` | float | Wind speed (m/s) |
| `voltage` | float | Panel output voltage (V) |
| `current` | float | Panel output current (A) |
| `GHI, DNI, DHI` | float | Solar irradiance components |
| `Quarter1–4 features` | float | Seasonal climate variables |

**Output:** `efficiency` ∈ [0.0, 1.0]

**Target Metrics:**
- R² Score: > 0.94
- Mean Absolute Error (MAE): minimized

---

### Model 2 — Site Suitability Classifier (XGBoost Classifier)

**Goal:** Binary classification — is a given location suitable for solar panel installation?

**Input Features:** 29 quarterly + annual climate variables including GHI, DNI, DHI, cloud cover, sunshine duration, ambient temperature, humidity, precipitation, and snowfall across all 4 quarters.

**Output:** `suitability` ∈ {`Yes`, `No`}

**Training Data:** Real-world Indian solar site dataset with labeled suitability outcomes.

---

### Training Data Overview

| Dataset | Rows | Key Columns | Source |
|---|---|---|---|
| `Solar_Sites_Dataset_India.csv` | ~500+ | GHI, DNI, DHI, quarterly weather, Label | Real Indian solar sites |
| `sensor_data.csv` | 5,000 | voltage, current, panel_temp, dust_index, efficiency | Synthetic (generated) |
| `solar_panel_combined_dataset.csv` | ~5,500+ | All merged features | Combined final dataset |
| `final_dataset.csv` | processed | Cleaned + engineered features | Pipeline output |

---

## 🗃️ Datasets

### Raw Data Sources

**`Solar_Sites_Dataset_India.csv`** — Real-world solar site evaluation data with quarterly meteorological variables and expert suitability labels. Each row represents a candidate installation site with features measuring climate viability for solar generation.

**`Wind_Sites_Dataset_India.csv`** — Wind turbine suitability data for Indian sites (used for future wind module expansion).

**`indian_weather_data.csv`** — Historical Indian weather records used to enrich the training dataset with real climatic conditions.

### Data Cleaning Strategies

```
CHALLENGE                          →  SOLUTION
─────────────────────────────────────────────────────────────────
Duplicate sensor entries           →  df.drop_duplicates()
Unrealistic sensor readings        →  Statistical outlier removal
Missing values                     →  df.fillna(0) + imputation
Sensor drift / noise               →  Normalization + smoothing
Weather ↔ sensor misalignment      →  Adaptive sampling + alignment
Noisy IoT data                     →  Anomaly filtering + synthetic features
Categorical text columns           →  pd.get_dummies(drop_first=True)
Label casing inconsistency         →  .map({'Yes':1,'No':0,'yes':1,'no':0})
```

---

## 🔍 Explainability with SHAP

SolarSense AI is built with **Explainable AI (XAI)** at its core. Engineers should never have to take an AI system's word for it — they need to understand *why* the model is raising an alarm.

**SHAP (SHapley Additive exPlanations)** decomposes each prediction into per-feature contributions grounded in cooperative game theory.

```
Example SHAP output for a low-efficiency prediction:
┌─────────────────────────────────────────────────────┐
│  Feature             │  SHAP Value  │  Direction     │
│─────────────────────────────────────────────────────│
│  panel_temp          │  +0.0510     │  ▲ Positive    │
│  temperature         │  +0.0161     │  ▲ Positive    │
│  DHI (% of GHI)      │  +0.0107     │  ▲ Positive    │
│  Quarter4-Precip     │  +0.0083     │  ▲ Positive    │
│  Quarter1-Cloud cover│  +0.0061     │  ▲ Positive    │
│  Quarter1-Sunshine   │  +0.0001     │  ▲ Positive    │
└─────────────────────────────────────────────────────┘
```

- **Positive SHAP value** → Feature *increases* predicted efficiency
- **Negative SHAP value** → Feature *decreases* predicted efficiency
- Displayed visually as a horizontal bar chart in the dashboard

---

## 🎨 Frontend Dashboard

The web dashboard (`main.html`) delivers an operator-grade interface in a single HTML file.

### UI Components

| Component | Technology | Purpose |
|---|---|---|
| **3D Solar Panel** | Three.js r128 | Real-time 3D model that animates based on prediction results |
| **Efficiency Gauge** | CSS + JS | Large numeric display with color coding (green/yellow/red) |
| **Risk Score Badge** | Bootstrap badges | Shows risk % with severity color (Low / Medium / High) |
| **SHAP Bar Chart** | Chart.js | Horizontal bar chart of top feature contributions |
| **Site Suitability Map** | Leaflet.js | Interactive geospatial map with site marker |
| **Cost Charts** | Chart.js | Monthly maintenance spend + cumulative annual expenditure |
| **AI Insights Panel** | HTML + API | Plain-language maintenance recommendations with icons |
| **Input Panel** | Bootstrap 5 | Sliders + number inputs for all 9 sensor/weather parameters |

### Design System

```css
:root {
  --primary:    #0d6efd;   /* Bootstrap blue — interactive elements */
  --dark-blue:  #1e3c72;   /* Deep navy — headers, hero section */
  --light-blue: #2a5298;   /* Mid blue — gradients */
  --accent:     #ffc107;   /* Amber — highlights, hero icon */
  --success:    #198754;   /* Green — healthy state indicators */
}
```

Design principles: **glassmorphism cards** (`backdrop-filter: blur`), floating hero animation, 12px border-radius everywhere, responsive grid, and animated 3D background canvas.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/PARTHDESHMUKH2005/solarsense-ai.git
cd solarsense-ai

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Train the ML Models

```bash
# Navigate to the ML source directory
cd ml/src

# Train both XGBoost models (generates .pkl files)
python train_ml.py
```

Expected output:
```
Dataset loaded successfully.
Processed dataframe shape: (5500, 42)

Training Efficiency Model...
Efficiency Model - R²: 0.9421 (Target >0.94), MAE: 0.0183

Training Suitability Model...
Suitability Model Accuracy: 0.9760
====================================
SUCCESS: All .pkl models generated!
====================================
```

### Run the Flask Server

```bash
# From ml/src directory (where .pkl files are located)
python app.py
```

```
================================================================================
SOLARSENSE AI - FLASK SERVER
================================================================================
Models loaded: ✓
SHAP available: ✓
OpenAI enabled: ⚠ Using rule-based insights

Server starting on http://localhost:5001
================================================================================
```

### Open the Dashboard

Navigate to **[http://localhost:5001](http://localhost:5001)** in your browser to access the full interactive dashboard.

### API Usage

```bash
# Test the prediction API directly
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "temperature": 42,
    "humidity": 65,
    "irradiance": 850,
    "dust_index": 0.7,
    "cloudcover": 10,
    "precip": 0,
    "wind_speed": 5,
    "voltage": 35,
    "current": 8
  }'
```

Example response:
```json
{
  "predicted_efficiency": 0.712,
  "risk_score": 28.8,
  "failure_flag": false,
  "explanation": {
    "panel_temp": 0.051,
    "temperature": 0.016,
    "dust_index": -0.043,
    "irradiance": 0.038
  },
  "insights_and_suggestions": "⚠ Good performance with room for optimization.\n🧹 Dust accumulation detected. Schedule cleaning to restore 5-10% efficiency.",
  "recommended_action": "Monitor closely - System performing well",
  "suitability": "Yes"
}
```

### Enable GPT-4 Insights (Optional)

In `ml/src/app.py`, set:

```python
USE_OPENAI = True
OPENAI_API_KEY = "your-openai-api-key-here"
```

---

## 📈 Performance Metrics

| Metric | Value |
|---|---|
| Efficiency Model R² Score | **> 0.94** |
| Site Suitability Accuracy | **> 97%** |
| Risk Score Range | 0–100% (continuous) |
| Failure Detection Threshold | Efficiency < 75% |
| API Response Time | ~50–120ms per prediction |
| Training Dataset Size | ~5,500 records |
| Feature Count (Efficiency Model) | 40+ engineered features |

---

## 🔮 Roadmap

```
Phase 1 — MVP (Complete ✅)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✅ XGBoost efficiency regressor
  ✅ Site suitability classifier
  ✅ SHAP explainability module
  ✅ Flask REST API
  ✅ Interactive HTML/JS dashboard
  ✅ 3D solar panel visualization
  ✅ Rule-based AI insights engine

Phase 2 — Enhanced Intelligence
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🔲 LSTM / Transformer time-series models for failure forecasting
  🔲 Self-learning models with real-time data feedback
  🔲 Live IoT sensor stream integration (MQTT/WebSocket)
  🔲 Edge AI deployment for on-site instant fault detection

Phase 3 — Multi-Energy Platform
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🔲 Wind turbine failure prediction module
  🔲 Battery storage State-of-Health (SOH) tracking
  🔲 Hybrid renewable park centralized monitoring
  🔲 ERP and workforce management system integration

Phase 4 — Intelligent Automation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🔲 Auto-generated maintenance schedules via AI
  🔲 Generative AI (GenAI) natural-language reports
  🔲 Chat-based assistant for engineers
  🔲 Satellite imagery analytics for long-term forecasting
  🔲 Mobile app with push notifications
```

---

## 👥 Team

<div align="center">

### 🌿 GreenMind AI

| Member | Role |
|:---:|:---:|
| **Keha** | Team Member |
| **Parth** | Team Member |
| **Arnav** | Team Member |
| **Lalit** | Team Member |

**🏛️ Institution:** Thapar Institute of Engineering and Technology

**🏆 Hackathon Problem Statement:** Predictive Maintenance for Renewable Infrastructure — *Create a model that predicts failures or efficiency drops in solar panels, wind turbines, or batteries using sensor data and weather patterns.*

</div>

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/YOUR_USERNAME/solarsense-ai.git

# Create your feature branch
git checkout -b feature/YourFeatureName

# Make changes and commit
git commit -m "feat: add YourFeatureName"

# Push and open a Pull Request
git push origin feature/YourFeatureName
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:ffc107,50:2a5298,100:1e3c72&height=140&section=footer" width="100%"/>

### ⚡ Powering the Future of Renewable Energy

**Made with 💚 for a sustainable planet by Team GreenMind AI**

*Thapar Institute of Engineering and Technology*

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-PARTHDESHMUKH2005/solarsense--ai-181717?style=for-the-badge&logo=github)](https://github.com/PARTHDESHMUKH2005/solarsense-ai)

⭐ **Star this repo if it helped you** — it motivates us to build more!

</div>
