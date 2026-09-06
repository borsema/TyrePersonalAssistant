def generate_alert(row):

    alerts = []

    # --------------------------
    # Pressure
    # --------------------------

    if row["pressure_psi"] < 26:

        alerts.append({
            "severity": "CRITICAL",
            "message":
                "Critical tyre pressure. "
                "Reduce driving and inspect immediately."
        })

    elif row["pressure_psi"] < 28:

        alerts.append({
            "severity": "WARNING",
            "message":
                f"Low tyre pressure ({row['pressure_psi']:.1f} PSI). "
                "Inflate to 32–34 PSI."
        })

    elif row["pressure_psi"] < 30:

        alerts.append({
            "severity": "WARNING",
            "message":
                f"Pressure slightly low ({row['pressure_psi']:.1f} PSI). "
                "Monitor closely."
        })

    # --------------------------
    # Slow leak detection
    # --------------------------

    if row["pressure_change"] < -0.3:

        alerts.append({
            "severity": "WARNING",
            "message":
                "Rapid pressure drop detected. "
                "Possible slow leak."
        })

    # --------------------------
    # Temperature
    # --------------------------

    if row["temperature_c"] > 45:

        alerts.append({
            "severity": "CRITICAL",
            "message":
                "Tyre overheating detected."
        })

    elif row["temperature_c"] > 38:

        alerts.append({
            "severity": "WARNING",
            "message":
                "Tyre temperature is higher than normal."
        })

    elif row["temperature_c"] > 33:

        alerts.append({
            "severity": "WARNING",
            "message":
                f"Tyre temperature elevated "
                f"({row['temperature_c']:.1f}°C). Monitor."
        })

    # --------------------------
    # Vibration
    # --------------------------

    if row["vibration"] > 0.70:

        alerts.append({
            "severity": "CRITICAL",
            "message":
                "Severe tyre vibration. "
                "Check for tyre damage immediately."
        })

    elif row["vibration"] > 0.50:

        alerts.append({
            "severity": "WARNING",
            "message":
                "High tyre vibration detected. "
                "Check tyre balance or road condition."
        })

    # --------------------------
    # ML prediction
    # --------------------------

    if row["health_status"] == "CRITICAL":

        alerts.append({
            "severity": "CRITICAL",
            "message":
                "ML model predicts critical tyre health."
        })

    elif row["health_status"] == "WARNING":

        alerts.append({
            "severity": "WARNING",
            "message":
                "ML model predicts tyre health risk."
        })

    # --------------------------
    # Remaining life
    # --------------------------

    if row["remaining_life_km"] < 2000:

        alerts.append({
            "severity": "CRITICAL",
            "message":
                f"Tyre replacement required immediately. "
                f"Only {row['remaining_life_km']:,.0f} km remaining."
        })

    elif row["remaining_life_km"] < 5000:

        alerts.append({
            "severity": "WARNING",
            "message":
                f"Low remaining tyre life: "
                f"{row['remaining_life_km']:,.0f} km."
        })

    # --------------------------
    # Environmental — Abrasion
    # --------------------------

    abrasion = float(row.get("abrasion_rate_mg_km", 0))

    if abrasion > 400:
        alerts.append({
            "severity": "CRITICAL",
            "message":
                f"Extreme tyre abrasion ({abrasion:.0f} mg/km). "
                "Reduce speed and check load immediately."
        })
    elif abrasion > 250:
        alerts.append({
            "severity": "WARNING",
            "message":
                f"High tyre abrasion rate ({abrasion:.0f} mg/km). "
                "Tyre wearing faster than normal."
        })

    # --------------------------
    # Environmental — Microplastics
    # --------------------------

    microplastic = float(row.get("microplastic_g_per_km", 0))

    if microplastic > 0.25:
        alerts.append({
            "severity": "CRITICAL",
            "message":
                f"Critical microplastic emission ({microplastic:.3f} g/km). "
                "Tyre shedding dangerous particle levels."
        })
    elif microplastic > 0.15:
        alerts.append({
            "severity": "WARNING",
            "message":
                f"Elevated microplastic shedding ({microplastic:.3f} g/km). "
                "Environmental impact is high."
        })

    # --------------------------
    # Environmental — PM2.5 Dust
    # --------------------------

    dust = float(row.get("dust_pm25_ug_m3", 0))

    if dust > 60:
        alerts.append({
            "severity": "CRITICAL",
            "message":
                f"Toxic PM2.5 at dangerous levels ({dust:.1f} µg/m³). "
                "Hazardous to health."
        })
    elif dust > 35:
        alerts.append({
            "severity": "WARNING",
            "message":
                f"High PM2.5 emission ({dust:.1f} µg/m³). "
                "Exceeds WHO safe air quality threshold."
        })

    if not alerts:

        alerts.append({
            "severity": "INFO",
            "message":
                "Tyre operating normally."
        })

    return alerts
