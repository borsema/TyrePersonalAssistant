import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

NUM_VEHICLES = 100
TYRE_POSITIONS = ["FL", "FR", "RL", "RR"]

OUTPUT_PATH = "data/tyre_historical_data.csv"

READINGS_PER_TYRE = 250
BASE_TYRE_LIFE_KM = 60000

os.makedirs("data", exist_ok=True)

records = []

start_time = datetime.now() - timedelta(days=30)

WEATHER_OPTIONS  = ["DRY", "WET", "RAIN", "SNOW"]
WEATHER_WEIGHTS  = [0.55, 0.25, 0.15, 0.05]

# ----------------------------------------------------------------
# ABRASION / EMISSION PHYSICS
# Baseline: ~80 mg/km for a new tyre at 60 km/h, dry road, 400 kg load
# Sources: EU tyre wear research, OECD microplastic estimates
# ----------------------------------------------------------------

def compute_env_metrics(speed_kmh, vehicle_load_kg, braking_intensity,
                        weather, wear_ratio, vibration):

    # Base abrasion rate mg/km
    base = 80.0

    # Speed factor — abrasion scales with speed^1.5
    speed_factor = (speed_kmh / 60.0) ** 1.5

    # Load factor — linear with load above reference 400 kg per tyre
    load_factor = 1.0 + max(0, (vehicle_load_kg - 400) / 400) * 0.5

    # Braking — hard braking dramatically increases abrasion
    brake_factor = 1.0 + braking_intensity * 2.5

    # Weather — wet/rain reduces abrasion slightly but increases
    # microplastic runoff; snow increases abrasion due to chains/grip
    weather_abrasion = {"DRY": 1.0, "WET": 0.85, "RAIN": 0.80, "SNOW": 1.30}
    w_factor = weather_abrasion.get(weather, 1.0)

    # Wear — worn tyres shed more particles (harder compound exposed)
    wear_factor = 1.0 + wear_ratio * 1.2

    # Vibration — rough road / imbalance increases abrasion
    vib_factor = 1.0 + max(0, vibration - 0.10) * 1.5

    abrasion_rate = base * speed_factor * load_factor * brake_factor * w_factor * wear_factor * vib_factor

    # Microplastics: ~60% of abrasion mass becomes particles < 5mm
    microplastic_g_per_km = abrasion_rate * 0.60 / 1000.0

    return (
        round(abrasion_rate, 3),
        round(microplastic_g_per_km, 5),
    )


for vehicle_num in range(1, NUM_VEHICLES + 1):

    vehicle_id = f"VEH_{vehicle_num:03d}"

    # Vehicle base load (kg per tyre) — varies by vehicle type
    base_load = np.random.uniform(300, 600)

    for tyre_position in TYRE_POSITIONS:

        tyre_id = f"{vehicle_id}_{tyre_position}"

        tyre_age_days = np.random.randint(30, 1825)
        manufacture_date = datetime.now() - timedelta(days=tyre_age_days)

        initial_distance = np.random.uniform(1000, 45000)

        scenario = np.random.choice(
            ["healthy", "slow_leak", "overheating", "high_vibration", "critical"],
            p=[0.50, 0.20, 0.12, 0.10, 0.08]
        )

        pressure    = np.random.uniform(33, 36)
        temperature = np.random.uniform(32, 45)
        vibration   = np.random.uniform(0.05, 0.20)
        distance_km = initial_distance

        for reading in range(READINGS_PER_TYRE):

            timestamp = start_time + timedelta(minutes=10 * reading)

            speed_kmh = np.clip(np.random.normal(65, 20), 0, 140)

            road_condition = np.random.choice(
                ["GOOD", "NORMAL", "ROUGH"], p=[0.45, 0.40, 0.15]
            )

            weather = np.random.choice(WEATHER_OPTIONS, p=WEATHER_WEIGHTS)

            # Load varies per interval (passengers, cargo changes)
            vehicle_load_kg = np.clip(
                base_load + np.random.normal(0, 50), 200, 800
            )

            # Braking intensity 0.0–1.0
            # Higher chance of hard braking at high speed or rough road
            brake_base = 0.05
            if speed_kmh > 80:
                brake_base += 0.10
            if road_condition == "ROUGH":
                brake_base += 0.10
            if weather in ("RAIN", "SNOW"):
                brake_base += 0.15
            braking_intensity = float(np.clip(
                np.random.exponential(brake_base), 0.0, 1.0
            ))

            distance_increment = speed_kmh / 6
            distance_km += distance_increment

            previous_pressure    = pressure
            previous_temperature = temperature

            if scenario == "healthy":
                pressure    = np.clip(pressure + np.random.normal(0, 0.25), 32, 36)
                temperature = np.clip(25 + speed_kmh * 0.25 + np.random.normal(0, 2), 25, 55)
                vibration   = np.clip(np.random.normal(0.12, 0.03), 0.03, 0.25)

            elif scenario == "slow_leak":
                pressure   -= np.random.uniform(0.02, 0.08)
                temperature = 28 + speed_kmh * 0.30 + np.random.normal(0, 3)
                vibration   = np.clip(np.random.normal(0.18, 0.05), 0.05, 0.40)

            elif scenario == "overheating":
                pressure   -= np.random.uniform(0.00, 0.03)
                temperature += np.random.uniform(0.05, 0.30) + speed_kmh * 0.05
                vibration   = np.clip(np.random.normal(0.20, 0.06), 0.05, 0.50)

            elif scenario == "high_vibration":
                pressure    += np.random.normal(0, 0.30)
                temperature  = 30 + speed_kmh * 0.30 + np.random.normal(0, 4)
                vibration   += np.random.uniform(0.01, 0.05)

            elif scenario == "critical":
                pressure   -= np.random.uniform(0.05, 0.15)
                temperature += np.random.uniform(0.10, 0.50)
                vibration  += np.random.uniform(0.02, 0.08)

            # Weather effect on pressure/temp
            if weather == "SNOW":
                temperature -= np.random.uniform(2, 5)
                pressure    -= np.random.uniform(0.1, 0.3)  # cold reduces pressure
            elif weather == "RAIN":
                temperature -= np.random.uniform(1, 3)

            # Load effect on temperature
            temperature += (vehicle_load_kg - 400) / 400 * 3.0

            # Braking effect on temperature
            temperature += braking_intensity * 8.0

            if road_condition == "ROUGH":
                vibration += np.random.uniform(0.03, 0.10)

            pressure    = np.clip(pressure, 10, 40)
            temperature = np.clip(temperature, 15, 120)
            vibration   = np.clip(vibration, 0.01, 1.50)

            pressure_change    = pressure - previous_pressure
            temperature_change = temperature - previous_temperature

            # Wear factor
            wear_factor = 1.0
            if pressure < 32:
                wear_factor += (32 - pressure) * 0.04
            if temperature > 55:
                wear_factor += (temperature - 55) * 0.01
            if vibration > 0.30:
                wear_factor += (vibration - 0.30) * 0.50
            if speed_kmh > 100:
                wear_factor += (speed_kmh - 100) * 0.005
            if road_condition == "ROUGH":
                wear_factor += 0.05
            if tyre_age_days > 1095:
                wear_factor += 0.15
            if braking_intensity > 0.5:
                wear_factor += braking_intensity * 0.20
            if weather == "SNOW":
                wear_factor += 0.10

            effective_usage = distance_km * wear_factor

            remaining_life_km = max(0, BASE_TYRE_LIFE_KM - effective_usage)
            remaining_life_km += np.random.normal(0, 500)
            remaining_life_km = max(0, min(remaining_life_km, BASE_TYRE_LIFE_KM))

            wear_ratio = min(1.0, distance_km / BASE_TYRE_LIFE_KM)

            # Environmental metrics
            abrasion_rate, microplastic_g_per_km = compute_env_metrics(
                speed_kmh, vehicle_load_kg, braking_intensity,
                weather, wear_ratio, vibration
            )

            # Health status
            risk_score = 0
            if pressure < 32:
                risk_score += (32 - pressure) * 8
            if temperature > 55:
                risk_score += (temperature - 55) * 1.2
            if vibration > 0.30:
                risk_score += (vibration - 0.30) * 30
            if remaining_life_km < 5000:
                risk_score += 30
            if scenario == "critical":
                risk_score += 20

            if risk_score < 10:
                health_status = "HEALTHY"
            elif risk_score < 25:
                health_status = "ATTENTION"
            elif risk_score < 50:
                health_status = "WARNING"
            else:
                health_status = "CRITICAL"

            records.append({
                "timestamp":             timestamp,
                "vehicle_id":            vehicle_id,
                "tyre_id":               tyre_id,
                "tyre_position":         tyre_position,
                "manufacture_date":      manufacture_date.date(),
                "tyre_age_days":         tyre_age_days,
                "distance_km":           round(distance_km, 2),
                "pressure_psi":          round(pressure, 2),
                "temperature_c":         round(temperature, 2),
                "vibration":             round(vibration, 3),
                "speed_kmh":             round(speed_kmh, 2),
                "road_condition":        road_condition,
                "weather_condition":     weather,
                "vehicle_load_kg":       round(vehicle_load_kg, 1),
                "braking_intensity":     round(braking_intensity, 3),
                "pressure_change":       round(pressure_change, 3),
                "temperature_change":    round(temperature_change, 3),
                "wear_ratio":            round(wear_ratio, 4),
                "abrasion_rate_mg_km":   abrasion_rate,
                "microplastic_g_per_km": microplastic_g_per_km,
                "tyre_health_status":    health_status,
                "remaining_life_km":     round(remaining_life_km, 2),
            })

df = pd.DataFrame(records)
df.to_csv(OUTPUT_PATH, index=False)

print("Tyre data generated successfully!")
print(f"Total records: {len(df):,}")
print("\nHealth Status Distribution:")
print(df["tyre_health_status"].value_counts())
print("\nEnvironmental Metrics Summary:")
print(df[["abrasion_rate_mg_km", "microplastic_g_per_km"]].describe())
print(f"\nData saved to: {OUTPUT_PATH}")
