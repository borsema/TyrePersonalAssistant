import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score

BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH      = os.path.join(BASE_DIR, "data", "tyre_historical_data.csv")
ENV_MODEL_PATH = os.path.join(BASE_DIR, "models", "tyre_env_model.pkl")

os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Features that drive environmental impact
env_features = [
    "speed_kmh",
    "vehicle_load_kg",
    "braking_intensity",
    "vibration",
    "wear_ratio",
    "temperature_c",
    "pressure_psi",
]

# Encode weather as numeric
weather_map = {"DRY": 0, "WET": 1, "RAIN": 2, "SNOW": 3}
df["weather_encoded"] = df["weather_condition"].map(weather_map).fillna(0)
env_features.append("weather_encoded")

# Encode road condition as numeric
road_map = {"GOOD": 0, "NORMAL": 1, "ROUGH": 2}
df["road_encoded"] = df["road_condition"].map(road_map).fillna(0)
env_features.append("road_encoded")

# Targets: 2 environmental outputs
env_targets = [
    "abrasion_rate_mg_km",
    "microplastic_g_per_km",
]

X = df[env_features]
y = df[env_targets]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("\n" + "=" * 60)
print("TRAINING TYRE ENVIRONMENTAL IMPACT MODEL")
print("=" * 60)

env_model = MultiOutputRegressor(
    RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
)

env_model.fit(X_train, y_train)
y_pred = env_model.predict(X_test)

for i, target in enumerate(env_targets):
    mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
    r2  = r2_score(y_test.iloc[:, i], y_pred[:, i])
    print(f"\n{target}:")
    print(f"  MAE: {mae:.5f}   R²: {r2:.4f}")

joblib.dump({
    "model":    env_model,
    "features": env_features,
    "targets":  env_targets,
    "weather_map": weather_map,
    "road_map":    road_map,
}, ENV_MODEL_PATH)

print(f"\nEnvironmental model saved to: {ENV_MODEL_PATH}")
