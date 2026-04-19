from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import os
import traceback

app = Flask(__name__)
CORS(app)

USE_OPENAI = False
OPENAI_API_KEY = "api"

# ─── Load Solar Models ─────────────────────────────────────────────────────
try:
    from explain import explain_prediction
    model_eff = joblib.load("efficiency_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    model_site = joblib.load("suitability_model.pkl")
    site_feature_columns = joblib.load("site_feature_columns.pkl")
    print("Models loaded successfully")
    print(f" Efficiency model features: {len(feature_columns)}")
    print(f" Suitability model features: {len(site_feature_columns)}")
except FileNotFoundError as e:
    print(f"ERROR: Could not load model files - {e}")
    exit(1)

# ─── Load Wind Turbine Models ──────────────────────────────────────────────
try:
    wind_model_eff         = joblib.load("wind_efficiency_model.pkl")
    wind_feature_columns   = joblib.load("wind_feature_columns.pkl")
    wind_model_site        = joblib.load("wind_suitability_model.pkl")
    wind_site_feature_cols = joblib.load("wind_site_feature_columns.pkl")
    WIND_MODELS_LOADED = True
    print("Wind turbine models loaded successfully")
except FileNotFoundError as e:
    WIND_MODELS_LOADED = False
    print(f"Wind turbine models not found - {e}")

# ─── SHAP ─────────────────────────────────────────────────────────────────
try:
    SHAP_AVAILABLE = True
    print(" SHAP explanation module loaded")
except ImportError:
    SHAP_AVAILABLE = False

    def explain_prediction(data):
        return {
            "temperature": -0.05 if data.get('temperature', 25) > 35 else 0.02,
            "humidity":    -0.03 if data.get('humidity', 50) > 70 else 0.01,
            "irradiance":   0.08 if data.get('irradiance', 800) > 700 else -0.05,
            "dust_index":  -0.06 if data.get('dust_index', 0.5) > 0.5 else 0.01,
        }

if USE_OPENAI:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        USE_OPENAI = False

# ─── Solar helpers ─────────────────────────────────────────────────────────
def get_genai_insights(shap_explanation, efficiency):
    if USE_OPENAI:
        try:
            prompt = f"Based on SHAP values {shap_explanation} for solar panel efficiency of {efficiency:.2%}. Provide 3 concise insights and actionable suggestions to improve or maintain efficiency."
            response = client.chat.completions.create(model="gpt-4", messages=[{"role":"user","content":prompt}], max_tokens=250, temperature=0.7)
            return response.choices[0].message.content.strip()
        except Exception as e:
            pass
    return get_fallback_insights(shap_explanation, efficiency)

def get_fallback_insights(shap_explanation, efficiency):
    insights = []
    if efficiency > 0.85:
        insights.append("✓ Excellent performance - System is operating at peak efficiency.")
    elif efficiency > 0.75:
        insights.append("⚠ Good performance with room for optimization.")
    else:
        insights.append("⚠ Low efficiency detected - Immediate attention required.")
    top_factors = sorted(shap_explanation.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    for feature, impact in top_factors:
        if 'temperature' in feature.lower() and impact < -0.03:
            insights.append("🌡️ High temperature is reducing efficiency. Consider cooling systems or shade structures.")
        elif 'irradiance' in feature.lower() and impact > 0.05:
            insights.append("☀️ Good solar irradiance levels. Maintain panel cleanliness to maximize capture.")
        elif 'dust' in feature.lower() and impact < -0.03:
            insights.append("🧹 Dust accumulation detected. Schedule cleaning to restore 5-10% efficiency.")
        elif 'humidity' in feature.lower() and impact < -0.02:
            insights.append("💧 High humidity affecting performance. Monitor for condensation issues.")
    if efficiency < 0.80:
        insights.append("📊 Recommendation: Conduct full system diagnostic and performance audit.")
    else:
        insights.append("🔄 Recommendation: Continue regular maintenance schedule for optimal performance.")
    return "\n".join(insights)

# ─── Wind helpers ──────────────────────────────────────────────────────────
def get_wind_explain(data):
    explanation = {}
    ws = data.get('wind_speed', 10)
    explanation['wind_speed']      = 0.08 * (ws / 12) if ws >= 3 else -0.10
    vib = data.get('vibration_level', 1.0)
    explanation['vibration_level'] = -0.06 * (vib - 1.0) if vib > 1.0 else 0.01
    gb = data.get('gearbox_temp', 70)
    explanation['gearbox_temp']    = -0.05 * ((gb - 80) / 20) if gb > 80 else 0.01
    gen = data.get('generator_temp', 75)
    explanation['generator_temp']  = -0.06 * ((gen - 90) / 20) if gen > 90 else 0.01
    brg = data.get('bearing_temp', 50)
    explanation['bearing_temp']    = -0.03 * ((brg - 70) / 15) if brg > 70 else 0.01
    hum = data.get('humidity', 50)
    explanation['humidity']        = -0.02 * ((hum - 70) / 30) if hum > 70 else 0.005
    return explanation

def get_wind_insights(explanation, efficiency):
    insights = []
    if efficiency > 0.40:
        insights.append("✓ Turbine operating at high efficiency — all systems nominal.")
    elif efficiency > 0.30:
        insights.append("⚠ Moderate efficiency — consider scheduled inspection.")
    else:
        insights.append("🔴 Low efficiency detected — immediate maintenance recommended.")
    top = sorted(explanation.items(), key=lambda x: abs(x[1]), reverse=True)[:4]
    for feat, val in top:
        if 'vibration' in feat and val < -0.03:
            insights.append("📳 High vibration detected. Inspect blade balance, tower bolts, and drivetrain.")
        elif 'gearbox_temp' in feat and val < -0.03:
            insights.append("⚙️ Gearbox overheating. Check lubrication levels and cooling fans immediately.")
        elif 'generator_temp' in feat and val < -0.03:
            insights.append("⚡ Generator temperature elevated. Inspect windings and cooling system.")
        elif 'bearing_temp' in feat and val < -0.02:
            insights.append("🔩 Bearing temperature high. Schedule greasing and bearing inspection.")
        elif 'wind_speed' in feat and val > 0.05:
            insights.append("💨 Excellent wind conditions — maximise generation by optimising yaw alignment.")
    if efficiency < 0.30:
        insights.append("📊 Recommendation: Take turbine offline for full diagnostic before next storm season.")
    else:
        insights.append("🔄 Recommendation: Log this reading and schedule routine maintenance per OEM schedule.")
    return "\n".join(insights)

def get_wind_power_output(wind_speed, efficiency):
    if wind_speed < 3 or wind_speed > 25:
        return 0.0
    air_density = 1.225
    rotor_radius = 40
    theoretical = 0.5 * air_density * 3.14159 * (rotor_radius ** 2) * (wind_speed ** 3) / 1000
    return min(theoretical * efficiency, 2000)

# ═══════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
def serve_html():
    try:
        return send_from_directory('.', 'static/main.html')
    except FileNotFoundError:
        return jsonify({"error": "main.html not found"}), 404

@app.route("/wind", methods=["GET"])
def serve_wind_html():
    try:
        return send_from_directory('.', 'static/wind.html')
    except FileNotFoundError:
        return jsonify({"error": "wind.html not found"}), 404

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "solar_models_loaded": True,
        "wind_models_loaded": WIND_MODELS_LOADED,
        "shap_available": SHAP_AVAILABLE,
        "openai_enabled": USE_OPENAI
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        required = ['temperature', 'humidity', 'irradiance']
        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({"error": f"Missing required fields: {missing}"}), 400
        if 'dust_index' not in data:
            data['dust_index'] = 0.5
        data['panel_temp']  = data['temperature'] + (data['irradiance'] / 800.0) * 20
        data['cloudcover']  = data.get('cloudcover', 0)
        data['precip']      = data.get('precip', 0)
        data['wind_speed']  = data.get('wind_speed', 5)
        data['voltage']     = data.get('voltage', 35)
        data['current']     = data.get('current', 8)
        df = pd.DataFrame([data]).reindex(columns=feature_columns, fill_value=0)
        efficiency = float(model_eff.predict(df)[0])
        efficiency = max(0.0, min(1.0, efficiency))
        risk_score   = round((1 - efficiency) * 100, 2)
        failure_flag = efficiency < 0.75
        try:
            explanation = explain_prediction(data)
        except Exception:
            explanation = {"error": "Explanation not available"}
        try:
            insights = get_genai_insights(explanation, efficiency)
        except Exception:
            insights = f"Efficiency: {efficiency:.1%}. System {'requires attention' if failure_flag else 'operating normally'}."
        try:
            site_data = {
                'GHI (kWh/m²/day)': data['irradiance'] / 200,
                'DNI (kWh/m²/day)': data['irradiance'] / 240,
                'DHI (% of GHI)': 20, 'Snowfall (mm/year)': 0,
                'Quarter1-Cloud cover': data['cloudcover'], 'Quarter1-Sunshine duration': 8,
                'Quarter1-Ambient temperature': data['temperature'], 'Quarter1-Relative humidity': data['humidity'],
                'Quarter1-Precipitation': data['precip'],
                'Quarter2-Cloud cover': data['cloudcover'], 'Quarter2-Sunshine duration': 8,
                'Quarter2-Ambient temperature': data['temperature'], 'Quarter2-Relative humidity': data['humidity'],
                'Quarter2-Precipitation': data['precip'],
                'Quarter3-Cloud cover': data['cloudcover'], 'Quarter3-Sunshine duration': 8,
                'Quarter3-Ambient temperature': data['temperature'], 'Quarter3-Relative humidity': data['humidity'],
                'Quarter3-Precipitation': data['precip'],
                'Quarter4-Cloud cover': data['cloudcover'], 'Quarter4-Sunshine duration': 8,
                'Quarter4-Ambient temperature': data['temperature'], 'Quarter4-Relative humidity': data['humidity'],
                'Quarter4-Precipitation': data['precip'],
                'YearlyCloud cover': data['cloudcover'], 'Sunshine duration': 8,
                'Ambient temperature': data['temperature'], 'Relative humidity': data['humidity'],
                'Precipitation': data['precip'] * 4,
            }
            df_site = pd.DataFrame([site_data]).reindex(columns=site_feature_columns, fill_value=0)
            suitability = 'Yes' if model_site.predict(df_site)[0] == 1 else 'No'
        except Exception:
            traceback.print_exc()
            suitability = 'Unknown'
        if risk_score < 30:
            action = "Monitor closely - System performing well"
        elif risk_score < 60:
            action = "Optimize - Consider maintenance and cleaning"
        else:
            action = "Immediate action required - Failure risk detected!"
        return jsonify({
            "predicted_efficiency": round(efficiency, 3),
            "risk_score": risk_score,
            "failure_flag": failure_flag,
            "explanation": explanation,
            "insights_and_suggestions": insights,
            "recommended_action": action,
            "suitability": suitability,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Prediction failed", "message": str(e), "traceback": traceback.format_exc()}), 500

@app.route("/predict_wind", methods=["POST"])
def predict_wind():
    if not WIND_MODELS_LOADED:
        return jsonify({"error": "Wind turbine models not loaded. Run gen_wind_models.py first."}), 503
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        required = ['wind_speed', 'gearbox_temp', 'generator_temp', 'vibration_level']
        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({"error": f"Missing required fields: {missing}"}), 400
        data.setdefault('wind_direction', 180)
        data.setdefault('blade_pitch',     5.0)
        data.setdefault('bearing_temp',   50.0)
        data.setdefault('ambient_temp',   25.0)
        data.setdefault('humidity',       55.0)
        data.setdefault('slope',          10.0)
        data.setdefault('elevation',     300.0)
        ws = data['wind_speed']
        data['turbine_rpm']          = ws * 4.0
        data['air_density']          = 1.225 - (data['ambient_temp'] - 15) * 0.004
        data['turbulence_intensity'] = data['vibration_level'] / max(ws, 0.1) * 100
        data['yearly_wind_speed']    = ws * 0.95
        df_eff = pd.DataFrame([data]).reindex(columns=wind_feature_columns, fill_value=0)
        efficiency = float(wind_model_eff.predict(df_eff)[0])
        efficiency = max(0.05, min(0.52, efficiency))
        power_kw     = round(get_wind_power_output(ws, efficiency), 1)
        risk_score   = round((1 - efficiency / 0.45) * 100, 2)
        risk_score   = max(0.0, min(100.0, risk_score))
        failure_flag = efficiency < 0.28
        explanation  = get_wind_explain(data)
        insights     = get_wind_insights(explanation, efficiency)
        try:
            site_input = {
                'wind_speed': ws, 'wind_direction': data['wind_direction'],
                'turbulence_intensity': data['turbulence_intensity'],
                'slope': data['slope'], 'elevation': data['elevation'],
                'humidity': data['humidity'], 'ambient_temp': data['ambient_temp'],
                'air_density': data['air_density'], 'yearly_wind_speed': data['yearly_wind_speed'],
            }
            df_site     = pd.DataFrame([site_input]).reindex(columns=wind_site_feature_cols, fill_value=0)
            suitability = 'Yes' if wind_model_site.predict(df_site)[0] == 1 else 'No'
        except Exception:
            traceback.print_exc()
            suitability = 'Unknown'
        if risk_score < 25:
            action = "Monitor — Turbine operating within healthy parameters"
        elif risk_score < 55:
            action = "Schedule inspection — One or more components showing early wear"
        else:
            action = "Immediate shutdown recommended — Critical failure risk detected!"
        capacity_factor = efficiency / 0.45 * 0.35
        daily_kwh       = round(power_kw * 24 * capacity_factor, 0)
        annual_mwh      = round(daily_kwh * 365 / 1000, 1)
        return jsonify({
            "predicted_efficiency":     round(efficiency, 3),
            "power_output_kw":          power_kw,
            "risk_score":               risk_score,
            "failure_flag":             failure_flag,
            "explanation":              explanation,
            "insights_and_suggestions": insights,
            "recommended_action":       action,
            "suitability":              suitability,
            "daily_energy_kwh":         daily_kwh,
            "annual_energy_mwh":        annual_mwh,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Wind prediction failed", "message": str(e)}), 500

if __name__ == "__main__":
    print("\n" + "="*80)
    print("RENEWSENSE AI — FLASK SERVER  (Solar + Wind Turbine)")
    print("="*80)
    print(f"Solar models      : OK")
    print(f"Wind models       : {'OK' if WIND_MODELS_LOADED else 'MISSING — run gen_wind_models.py'}")
    print(f"SHAP              : {'OK' if SHAP_AVAILABLE else 'fallback'}")
    print(f"Routes: / (solar)   /wind (turbine)   /predict   /predict_wind")
    print(f"\nServer → http://localhost:5001")
    print("="*80 + "\n")

app.run(debug=False, port=int(os.environ.get("PORT", 5001)), host='0.0.0.0')
