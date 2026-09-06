import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEALTH_MODEL_PATH = os.path.join(BASE_DIR, "models", "tyre_health_model.pkl")
RUL_MODEL_PATH = os.path.join(BASE_DIR, "models", "tyre_rul_model.pkl")
ENV_MODEL_PATH = os.path.join(BASE_DIR, "models", "tyre_env_model.pkl")

# ---------------------------------------
# LOAD MODELS
# ---------------------------------------

health_package = joblib.load(
    HEALTH_MODEL_PATH
)

rul_package = joblib.load(
    RUL_MODEL_PATH
)

env_package = joblib.load(
    ENV_MODEL_PATH
)

health_model = health_package["model"]
label_encoder = health_package["label_encoder"]
features = health_package["features"]

rul_model = rul_package["model"]

env_model = env_package["model"]
env_features = env_package["features"]
env_targets = env_package["targets"]
weather_map = env_package["weather_map"]
road_map = env_package["road_map"]

# ---------------------------------------
# NEW INCOMING TYRE DATA
# ---------------------------------------

new_tyre_data = {
    "tyre_age_days": 800,
    "distance_km": 38000,
    "pressure_psi": 29.5,
    "temperature_c": 62.0,
    "vibration": 0.48,
    "speed_kmh": 75.0,
    "pressure_change": -1.2,
    "temperature_change": 3.5,
    "vehicle_load_kg": 450.0,
    "braking_intensity": 0.35,
    "wear_ratio": 0.52,
    "weather_condition": "WET",
    "road_condition": "NORMAL"
}

# Convert to DataFrame
input_df = pd.DataFrame(
    [new_tyre_data]
)

# Ensure correct feature order
input_df = input_df[features]

# ---------------------------------------
# HEALTH PREDICTION
# ---------------------------------------

health_prediction = health_model.predict(
    input_df
)

health_status = label_encoder.inverse_transform(
    health_prediction
)[0]

health_probability = health_model.predict_proba(
    input_df
)[0]

# ---------------------------------------
# RUL PREDICTION
# ---------------------------------------

remaining_life_km = rul_model.predict(
    input_df
)[0]

remaining_life_km = max(
    0,
    remaining_life_km
)

# ---------------------------------------
# DISPLAY RESULTS
# ---------------------------------------

print("\n" + "=" * 60)
print("🛞 TPA - TYRE PERSONAL ASSISTANT")
print("=" * 60)

print("\nINCOMING SENSOR DATA")

for key, value in new_tyre_data.items():
    print(f"{key}: {value}")

print("\n" + "-" * 60)

print("HEALTH PREDICTION")

print(f"Status: {health_status}")

print("\nClass Probabilities:")

for label, probability in zip(
    label_encoder.classes_,
    health_probability
):
    print(
        f"{label}: {probability * 100:.2f}%"
    )

print("\n" + "-" * 60)

print("REMAINING USEFUL LIFE PREDICTION")

print(
    f"Estimated Remaining Life: "
    f"{remaining_life_km:,.0f} km"
)

print("\n" + "-" * 60)

# ---------------------------------------
# SIMPLE RECOMMENDATION
# ---------------------------------------

if health_status == "HEALTHY":
    recommendation = (
        "Tyre is operating normally. "
        "Continue regular monitoring."
    )

elif health_status == "ATTENTION":
    recommendation = (
        "Monitor the tyre closely and "
        "inspect pressure during the next service."
    )

elif health_status == "WARNING":
    recommendation = (
        "Tyre requires inspection soon. "
        "Check pressure, temperature and physical condition."
    )

else:
    recommendation = (
        "CRITICAL: Reduce driving and "
        "inspect or replace the tyre immediately."
    )

print("\nTPA RECOMMENDATION")

print(recommendation)

print("\n" + "=" * 60)

# ---------------------------------------
# ENVIRONMENTAL IMPACT PREDICTION
# ---------------------------------------

env_input = pd.DataFrame([new_tyre_data])

# Encode weather and road conditions
env_input["weather_encoded"] = (
    env_input["weather_condition"].map(weather_map).fillna(0)
)
env_input["road_encoded"] = (
    env_input["road_condition"].map(road_map).fillna(0)
)

# Select env model features in correct order
env_input_df = env_input[env_features]

# Predict environmental impact
env_predictions = env_model.predict(env_input_df)[0]

# Build results dict
env_results = {
    target: max(0, pred)
    for target, pred in zip(env_targets, env_predictions)
}

# ---------------------------------------
# DISPLAY ENV RESULTS
# ---------------------------------------

print("\n" + "=" * 60)
print("🌍 TPA - ENVIRONMENTAL IMPACT PREDICTION")
print("=" * 60)

print("\nDRIVING CONDITIONS")
print(f"Weather Condition : {new_tyre_data['weather_condition']}")
print(f"Road Condition    : {new_tyre_data['road_condition']}")
print(f"Speed             : {new_tyre_data['speed_kmh']} km/h")
print(f"Vehicle Load      : {new_tyre_data['vehicle_load_kg']} kg")
print(f"Braking Intensity : {new_tyre_data['braking_intensity']}")

print("\n" + "-" * 60)
print("PREDICTED ENVIRONMENTAL OUTPUT (per km)")

print(
    f"Abrasion Rate     : "
    f"{env_results['abrasion_rate_mg_km']:.4f} mg/km"
)
print(
    f"Microplastic Shed : "
    f"{env_results['microplastic_g_per_km']:.6f} g/km"
)

print("\n" + "-" * 60)

# Simple environmental rating based on abrasion rate
abrasion = env_results["abrasion_rate_mg_km"]
if abrasion < 5:
    env_rating = "LOW IMPACT    ✅"
    env_advice = (
        "Tyre wear is within normal limits. "
        "Minimal environmental footprint."
    )
elif abrasion < 10:
    env_rating = "MODERATE IMPACT  ⚠️"
    env_advice = (
        "Moderate tyre wear detected. Consider smoother "
        "driving and checking tyre pressure."
    )
else:
    env_rating = "HIGH IMPACT   🔴"
    env_advice = (
        "High abrasion rate detected. Inspect tyre condition, "
        "reduce speed, and avoid aggressive braking."
    )

print("ENVIRONMENTAL RATING")
print(f"Rating : {env_rating}")
print(f"Advice : {env_advice}")

print("\n" + "=" * 60)