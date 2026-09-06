import random
from datetime import datetime

import numpy as np
import pandas as pd


TOTAL_TYRE_LIFE_KM  = 60000
TYRE_POSITIONS      = ["FL", "FR", "RL", "RR"]
SIMULATION_MULTIPLIER = 50

WEATHER_OPTIONS = ["DRY", "WET", "RAIN", "SNOW"]
WEATHER_WEIGHTS = [0.55, 0.25, 0.15, 0.05]


# ============================================================
# ABRASION / EMISSION PHYSICS  (mirrors generate_data.py)
# ============================================================

def compute_env_metrics(speed_kmh, vehicle_load_kg, braking_intensity,
                        weather, wear_ratio, vibration):

    base         = 80.0
    speed_factor = (max(speed_kmh, 1) / 60.0) ** 1.5
    load_factor  = 1.0 + max(0, (vehicle_load_kg - 400) / 400) * 0.5
    brake_factor = 1.0 + braking_intensity * 2.5
    weather_abrasion = {"DRY": 1.0, "WET": 0.85, "RAIN": 0.80, "SNOW": 1.30}
    w_factor     = weather_abrasion.get(weather, 1.0)
    wear_factor  = 1.0 + wear_ratio * 1.2
    vib_factor   = 1.0 + max(0, vibration - 0.10) * 1.5

    abrasion_rate        = base * speed_factor * load_factor * brake_factor * w_factor * wear_factor * vib_factor
    microplastic_g_per_km = abrasion_rate * 0.60 / 1000.0

    return (
        round(abrasion_rate, 3),
        round(microplastic_g_per_km, 5),
    )


# ============================================================
# INITIALIZE NEW TYRES
# ============================================================

def initialize_new_tyres():

    weather = random.choices(WEATHER_OPTIONS, weights=WEATHER_WEIGHTS, k=1)[0]
    vehicle_load_kg = round(random.uniform(350, 550), 1)

    data = []

    for position in TYRE_POSITIONS:

        abrasion, microplastic = compute_env_metrics(
            0, vehicle_load_kg, 0.0, weather, 0.0, 0.10
        )

        data.append({
            "timestamp":             datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tyre_position":         position,
            "tyre_age_km":           0.0,
            "remaining_life_km":     float(TOTAL_TYRE_LIFE_KM),
            "pressure_psi":          round(random.uniform(32.0, 34.0), 2),
            "temperature_c":         round(random.uniform(25.0, 30.0), 2),
            "vibration":             round(random.uniform(0.05, 0.15), 3),
            "speed_kmh":             0.0,
            "vehicle_load_kg":       vehicle_load_kg,
            "braking_intensity":     0.0,
            "weather_condition":     weather,
            "pressure_change":       0.0,
            "temperature_change":    0.0,
            "abrasion_rate_mg_km":   abrasion,
            "microplastic_g_per_km": microplastic,
            "simulation_step":       0,
        })

    return pd.DataFrame(data)


# ============================================================
# CALCULATE DISTANCE
# ============================================================

def calculate_distance(speed_kmh):
    return round(speed_kmh * (10 / 60) * SIMULATION_MULTIPLIER, 2)


# ============================================================
# GENERATE PRESSURE
# ============================================================

def generate_pressure(current_pressure, tyre_age_km, weather, vehicle_load_kg):

    pressure_change = random.uniform(-0.4, 0.4)

    age_ratio = tyre_age_km / TOTAL_TYRE_LIFE_KM
    if random.random() < age_ratio:
        pressure_change -= random.uniform(0.02, 0.12)

    # Cold weather reduces pressure
    if weather == "SNOW":
        pressure_change -= random.uniform(0.1, 0.3)
    elif weather == "RAIN":
        pressure_change -= random.uniform(0.0, 0.1)

    # Heavy load slightly increases pressure
    pressure_change += (vehicle_load_kg - 400) / 4000

    new_pressure = max(20.0, min(38.0, current_pressure + pressure_change))
    return round(new_pressure, 2), round(pressure_change, 3)


# ============================================================
# GENERATE TEMPERATURE
# ============================================================

def generate_temperature(current_temperature, speed, tyre_age_km,
                         vehicle_load_kg, braking_intensity, weather):

    speed_heat   = speed / 20
    age_heat     = (tyre_age_km / TOTAL_TYRE_LIFE_KM) * 5
    load_heat    = (vehicle_load_kg - 400) / 400 * 3.0
    brake_heat   = braking_intensity * 8.0

    weather_cool = {"DRY": 0, "WET": -2, "RAIN": -3, "SNOW": -5}
    w_cool       = weather_cool.get(weather, 0)

    target_temperature = (
        25 + speed_heat + age_heat + load_heat + brake_heat
        + w_cool + random.uniform(-3, 3)
    )

    new_temperature    = current_temperature * 0.5 + target_temperature * 0.5
    temperature_change = new_temperature - current_temperature

    return round(new_temperature, 2), round(temperature_change, 3)


# ============================================================
# GENERATE VIBRATION
# ============================================================

def generate_vibration(tyre_age_km):
    wear_ratio = tyre_age_km / TOTAL_TYRE_LIFE_KM
    vibration  = 0.10 + wear_ratio * 0.70 + random.uniform(-0.05, 0.10)
    return round(max(0.01, vibration), 3)


# ============================================================
# NEXT 10 MINUTE SIMULATION
# ============================================================

def simulate_next_interval(current_state):

    vehicle_speed   = random.uniform(30, 100)
    distance_travelled = calculate_distance(vehicle_speed)

    # Shared interval parameters
    weather         = random.choices(WEATHER_OPTIONS, weights=WEATHER_WEIGHTS, k=1)[0]
    vehicle_load_kg = round(random.uniform(300, 650), 1)

    # Braking intensity — influenced by speed and weather
    brake_base = 0.05
    if vehicle_speed > 80:
        brake_base += 0.10
    if weather in ("RAIN", "SNOW"):
        brake_base += 0.15
    braking_intensity = round(float(np.clip(
        np.random.exponential(brake_base), 0.0, 1.0
    )), 3)

    new_data = []

    for _, row in current_state.iterrows():

        current_age  = float(row["tyre_age_km"])
        new_age      = min(TOTAL_TYRE_LIFE_KM, current_age + distance_travelled)
        remaining_life = max(0.0, TOTAL_TYRE_LIFE_KM - new_age)
        wear_ratio   = min(1.0, new_age / TOTAL_TYRE_LIFE_KM)

        pressure, pressure_change = generate_pressure(
            float(row["pressure_psi"]), new_age, weather, vehicle_load_kg
        )

        temperature, temperature_change = generate_temperature(
            float(row["temperature_c"]), vehicle_speed, new_age,
            vehicle_load_kg, braking_intensity, weather
        )

        vibration = generate_vibration(new_age)

        abrasion, microplastic = compute_env_metrics(
            vehicle_speed, vehicle_load_kg, braking_intensity,
            weather, wear_ratio, vibration
        )

        new_data.append({
            "timestamp":             datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tyre_position":         row["tyre_position"],
            "tyre_age_km":           round(new_age, 2),
            "remaining_life_km":     round(remaining_life, 2),
            "pressure_psi":          pressure,
            "temperature_c":         temperature,
            "vibration":             vibration,
            "speed_kmh":             round(vehicle_speed, 2),
            "vehicle_load_kg":       vehicle_load_kg,
            "braking_intensity":     braking_intensity,
            "weather_condition":     weather,
            "pressure_change":       pressure_change,
            "temperature_change":    temperature_change,
            "abrasion_rate_mg_km":   abrasion,
            "microplastic_g_per_km": microplastic,
            "simulation_step":       int(row["simulation_step"]) + 1,
        })

    return pd.DataFrame(new_data)
