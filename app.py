import os
import sys
import urllib.request

# Ensure project root is on path (required on Streamlit Cloud)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from datetime import datetime

from src.live_simulator import (
    initialize_new_tyres,
    simulate_next_interval
)

from src.predict_live import TyrePredictor
from src.predict_env import EnvPredictor
from src.alert_engine import generate_alert


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="TPA | Tyre Personal Assistant",
    page_icon="🛞",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# MODEL DOWNLOAD (Streamlit Cloud — LFS fallback)
# Raw download URLs for each model stored in GitHub LFS
# ============================================================

_REPO_RAW = (
    "https://media.githubusercontent.com/media/"
    "borsema/TyrePersonalAssistant/main/models"
)

_MODELS = {
    "models/tyre_health_model.pkl": f"{_REPO_RAW}/tyre_health_model.pkl",
    "models/tyre_rul_model.pkl":    f"{_REPO_RAW}/tyre_rul_model.pkl",
    "models/tyre_env_model.pkl":    f"{_REPO_RAW}/tyre_env_model.pkl",
}

os.makedirs("models", exist_ok=True)

_missing = [
    (path, url) for path, url in _MODELS.items()
    if not os.path.exists(path) or os.path.getsize(path) < 512
]

if _missing:
    with st.spinner(
        f"⏳ Downloading model files ({len(_missing)}/3)…  "
        "This only happens once on first deploy."
    ):
        for _path, _url in _missing:
            try:
                urllib.request.urlretrieve(_url, _path)
            except Exception as e:
                st.error(
                    f"❌ Failed to download `{os.path.basename(_path)}`:\n\n{e}"
                    "\n\nEnsure the repository is public and models are pushed via Git LFS."
                )
                st.stop()
    st.rerun()


# ============================================================
# LOAD ML MODEL
# ============================================================

@st.cache_resource
def load_predictor():
    return TyrePredictor()

@st.cache_resource
def load_env_predictor():
    return EnvPredictor()

predictor     = load_predictor()
env_predictor = load_env_predictor()


# ============================================================
# INITIALIZE NEW SIMULATION SESSION
# ============================================================

if "live_tyre_data" not in st.session_state:

    st.session_state.live_tyre_data = (
        initialize_new_tyres()
    )

    st.session_state.simulation_count = 0


# ============================================================
# INITIAL MODEL PREDICTION
# ============================================================

def prepare_for_prediction(df):
    out = df.copy()
    out["distance_km"]   = out["tyre_age_km"]
    out["tyre_age_days"] = (out["tyre_age_km"] / 50).round(2)
    if "wear_ratio" not in out.columns:
        out["wear_ratio"] = (out["tyre_age_km"] / 60000).clip(0, 1)
    if "vehicle_load_kg" not in out.columns:
        out["vehicle_load_kg"] = 400.0
    if "braking_intensity" not in out.columns:
        out["braking_intensity"] = 0.05
    return out


if "predictions" not in st.session_state:

    st.session_state.predictions = (
        predictor.predict(
            prepare_for_prediction(
                st.session_state.live_tyre_data
            )
        )
    )
    st.session_state.env_predictions = (
        env_predictor.predict(
            st.session_state.live_tyre_data
        )
    )


# ============================================================
# GET CURRENT PREDICTIONS
# ============================================================

predictions = (
    st.session_state.predictions.copy()
)

env_preds = st.session_state.get("env_predictions", None)
if env_preds is not None:
    for col in ["abrasion_rate_mg_km", "microplastic_g_per_km"]:
        if col in env_preds.columns:
            predictions[col] = env_preds[col].values


# ============================================================
# SIMULATION BUTTON
# ============================================================

top1, top2, top3 = st.columns([1, 2, 1])

st.markdown("""
<style>
div.stButton > button {
    background: linear-gradient(135deg, #0d3a6e, #1570cc) !important;
    color: #e0f0ff !important;
    border: 1px solid #39a6ff !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    font-weight: bold !important;
    letter-spacing: 2px !important;
    padding: 6px 48px !important;
    box-shadow: 0 0 12px rgba(57,166,255,0.3) !important;
    transition: all 0.2s ease !important;
    margin-top: 0px !important;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #1570cc, #39a6ff) !important;
    box-shadow: 0 0 20px rgba(57,166,255,0.6) !important;
    color: #fff !important;
}
</style>
""", unsafe_allow_html=True)

# ── Simulation info note ──────────────────────────────
_sim_speed  = float(st.session_state.live_tyre_data["speed_kmh"].mean())
_sim_dist   = round(_sim_speed * (10 / 60) * 50, 1)
_total_dist = float(st.session_state.live_tyre_data["tyre_age_km"].mean())

with top2:
    _, btn_col, _ = st.columns([0.5, 2, 0.5])
    with btn_col:
        if st.button(
            "⚡ SIMULATE NEXT 10 MIN",
            use_container_width=True
        ):

            # Generate the next tyre state
            new_data = simulate_next_interval(
                st.session_state.live_tyre_data
            )

            # Save the updated continuous state
            st.session_state.live_tyre_data = (
                new_data
            )

            # Predict updated tyre condition
            st.session_state.predictions = (
                predictor.predict(
                    prepare_for_prediction(new_data)
                )
            )
            st.session_state.env_predictions = (
                env_predictor.predict(new_data)
            )

            # Increase simulation time
            st.session_state.simulation_count += 1

            # Update dashboard
            st.rerun()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_tyre(position):

    return predictions[
        predictions["tyre_position"] == position
    ].iloc[0]


def get_status_color(status):

    return {
        "HEALTHY": "#21e36d",
        "ATTENTION": "#ffd23f",
        "WARNING": "#ff9f1c",
        "CRITICAL": "#ff3b3b"
    }.get(
        str(status).upper(),
        "#21e36d"
    )


def get_overall_status():

    statuses = (
        predictions["health_status"]
        .astype(str)
        .str.upper()
    )

    if "CRITICAL" in statuses.values:

        return (
            "CRITICAL",
            "#ff3b3b",
            "Immediate Tyre Inspection Required"
        )

    if "WARNING" in statuses.values:

        return (
            "WARNING",
            "#ff9f1c",
            "Drive Safe. Check Tyre Condition."
        )

    if "ATTENTION" in statuses.values:

        return (
            "ATTENTION",
            "#ffd23f",
            "Monitor Tyre Conditions"
        )

    return (
        "ALL CLEAR",
        "#21e36d",
        "All Tyres Operating Normally"
    )


# ============================================================
# GET INDIVIDUAL TYRES
# ============================================================

fl = get_tyre("FL")
fr = get_tyre("FR")
rl = get_tyre("RL")
rr = get_tyre("RR")


# ============================================================
# VEHICLE METRICS
# ============================================================

speed = float(
    min(
        max(
            predictions["speed_kmh"].mean(),
            0
        ),
        180
    )
)


rpm = min(
    8.0,
    0.8 + speed / 20
)


avg_temp = float(
    predictions["temperature_c"].mean()
)


# ============================================================
# HEALTH SCORE
# ============================================================

health_map = {
    "HEALTHY": 100,
    "ATTENTION": 75,
    "WARNING": 50,
    "CRITICAL": 20
}


predictions["health_score"] = (

    predictions["health_status"]
    .astype(str)
    .str.upper()
    .map(health_map)
    .fillna(100)
)


overall_health = float(
    predictions["health_score"].mean()
)


# ============================================================
# WORST TYRE / REMAINING LIFE
# ============================================================

priority_map = {
    "CRITICAL": 1,
    "WARNING": 2,
    "ATTENTION": 3,
    "HEALTHY": 4
}


predictions["priority"] = (

    predictions["health_status"]
    .astype(str)
    .str.upper()
    .map(priority_map)
    .fillna(5)
)


worst_tyre = (

    predictions
    .sort_values(
        [
            "priority",
            "remaining_life_km"
        ]
    )
    .iloc[0]
)


remaining_life = max(
    0,
    float(
        worst_tyre["remaining_life_km"]
    )
)


life_percentage = min(
    100,
    (
        remaining_life / 60000
    ) * 100
)


# ============================================================
# STATUS COUNTS
# ============================================================

status_counts = {
    "HEALTHY": 0,
    "ATTENTION": 0,
    "WARNING": 0,
    "CRITICAL": 0
}


for s in (
    predictions["health_status"]
    .astype(str)
    .str.upper()
):

    if s in status_counts:

        status_counts[s] += 1


# ============================================================
# GENERATE ALERTS
# ============================================================

alerts = []


for _, tyre in predictions.iterrows():

    try:

        for alert in generate_alert(tyre):

            if alert.get("severity") != "INFO":

                alerts.append({

                    "position":
                        tyre["tyre_position"],

                    "severity":
                        alert.get("severity"),

                    "message":
                        alert.get("message")
                })

    except Exception:
        pass


# ============================================================
# OVERALL STATUS
# ============================================================

overall_status, overall_color, overall_message = (
    get_overall_status()
)


# ============================================================
# ============================================================
# TIME / SIMULATION
# ============================================================

current_time = datetime.now().strftime(
    "%I:%M %p"
)


simulation_minutes = (
    st.session_state.simulation_count * 10
)


# ============================================================
# TYRE DATA FORMATTER
# ============================================================

def tyre_dict(tyre):

    status = str(
        tyre["health_status"]
    ).upper()


    return {
        "pressure":    float(tyre["pressure_psi"]),
        "temperature": float(tyre["temperature_c"]),
        "status":      status,
        "color":       get_status_color(status),
        "rul":         float(tyre["remaining_life_km"]),
        "abrasion":    float(tyre.get("abrasion_rate_mg_km", 0)),
        "microplastic":float(tyre.get("microplastic_g_per_km", 0)),
        "weather":     str(tyre.get("weather_condition", "DRY")),
        "load":        float(tyre.get("vehicle_load_kg", 400)),
        "braking":     float(tyre.get("braking_intensity", 0)),
    }


fl_data = tyre_dict(fl)
fr_data = tyre_dict(fr)
rl_data = tyre_dict(rl)
rr_data = tyre_dict(rr)


# ============================================================
# TYRE PRESSURE ARC
# ============================================================

def tyre_arc(
    value,
    max_val,
    color,
    size=120
):

    pct = min(
        value / max_val,
        1.0
    )


    r = 44

    circ = (
        2 * 3.14159 * r
    )


    dash = (
        pct *
        circ *
        0.75
    )


    gap = circ


    return f'''
    <svg
        width="{size}"
        height="{size}"
        viewBox="0 0 100 100"
    >

        <circle
            cx="50"
            cy="50"
            r="{r}"
            fill="none"
            stroke="#1a2a3a"
            stroke-width="9"
        />

        <circle
            cx="50"
            cy="50"
            r="{r}"
            fill="none"
            stroke="{color}"
            stroke-width="9"
            stroke-dasharray="{dash:.1f} {gap:.1f}"
            stroke-linecap="round"
            transform="rotate(-225 50 50)"
        />

    </svg>
    '''


# ============================================================
# STATUS BADGE
# ============================================================

def status_badge(
    status,
    color
):

    return f'''

    <div style="
        display:inline-block;
        background:{color}22;
        border:1px solid {color};
        color:{color};
        border-radius:6px;
        padding:3px 10px;
        font-size:11px;
        font-weight:bold;
        letter-spacing:1px;
        margin-top:6px;
    ">

        {status}

    </div>
    '''


# ============================================================
# REMAINING LIFE CARD
# ============================================================

# Avg env metrics across all 4 tyres
avg_abrasion    = float(predictions.get("abrasion_rate_mg_km",    pd.Series([0])).mean()) if "abrasion_rate_mg_km"    in predictions.columns else 0.0
avg_microplastic= float(predictions.get("microplastic_g_per_km", pd.Series([0])).mean()) if "microplastic_g_per_km" in predictions.columns else 0.0
weather_now     = str(predictions["weather_condition"].iloc[0]) if "weather_condition" in predictions.columns else "DRY"



def rul_card(
    label,
    tyre_d
):

    rul = max(
        0,
        tyre_d["rul"]
    )


    pct = min(
        100,
        (
            rul / 60000
        ) * 100
    )


    color = tyre_d["color"]


    return f'''

    <div class="rul-item">

        <div class="rul-pos">
            {label}
        </div>

        <div
            class="rul-val"
            style="color:{color};"
        >

            {rul:,.0f}

            <span class="rul-km">
                km
            </span>

        </div>

        <div class="rul-bar">

            <div
                class="rul-marker"
                style="left:{pct:.1f}%;">
            </div>

        </div>

    </div>
    '''


# ============================================================
# BUILD RUL CARDS
# ============================================================

rul_cards_html = (

    rul_card(
        "FRONT LEFT",
        fl_data
    )

    +

    rul_card(
        "FRONT RIGHT",
        fr_data
    )

    +

    rul_card(
        "REAR LEFT",
        rl_data
    )

    +

    rul_card(
        "REAR RIGHT",
        rr_data
    )
)


# ============================================================
# BUILD ALERT HTML
# ============================================================

if len(alerts) == 0:

    alert_html = '''

    <div style="
        text-align:center;
        margin-top:20px;
        color:#21e36d;
        font-size:13px;
        font-weight:bold;
    ">

        ● ALL TYRES HEALTHY

        <br>

        <span style="
            color:#6a8a9a;
            font-size:11px;
            font-weight:normal;
        ">

            No issues detected

        </span>

    </div>
    '''

else:

    # Group alerts by tyre position
    grouped = {}
    for alert in alerts:
        pos = alert["position"]
        if pos not in grouped:
            grouped[pos] = []
        grouped[pos].append(alert)

    alert_html = ""

    for pos, pos_alerts in grouped.items():

        # Highest severity for this tyre
        sev = "CRITICAL" if any(a["severity"] == "CRITICAL" for a in pos_alerts) else "WARNING"

        color = "#ff3b3b" if sev == "CRITICAL" else "#ff9f1c"
        icon = "🔴" if sev == "CRITICAL" else "⚠️"

        # Combine all messages as bullet points
        messages = "".join(
            f'<div style="color:#7a90a4;font-size:13px;margin-top:3px;">• {a["message"]}</div>'
            for a in pos_alerts
        )

        alert_html += f'''
        <div style="
            display:flex;
            gap:10px;
            padding:10px 0;
            border-bottom:1px solid rgba(255,255,255,0.06);
            align-items:flex-start;
        ">
            <div style="font-size:16px;margin-top:2px;">{icon}</div>
            <div style="flex:1;">
                <div style="color:{color};font-size:14px;font-weight:bold;">
                    {pos} Tyre &mdash; {sev}
                </div>
                {messages}
            </div>
            <div style="color:#4a6a7a;font-size:12px;white-space:nowrap;">{current_time}</div>
        </div>
        '''


# ============================================================
# BUILD ENV ALERT HTML
# ============================================================

env_alert_rows = []

if avg_microplastic > 0.25:
    env_alert_rows.append(("🔴", "#ff3b3b", "CRITICAL", f"Microplastics {avg_microplastic:.4f} g/km — exceeds 0.25 g/km limit"))
elif avg_microplastic > 0.15:
    env_alert_rows.append(("⚠️", "#ff9f1c", "WARNING", f"Microplastics {avg_microplastic:.4f} g/km — above 0.15 g/km threshold"))

if env_alert_rows:
    env_alert_html = "".join(f'''
    <div style="display:flex;gap:10px;padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.06);align-items:center;">
        <div style="font-size:15px;">{icon}</div>
        <div style="flex:1;">
            <div style="color:{color};font-size:13px;font-weight:bold;">{sev}</div>
            <div style="color:#7a90a4;font-size:13px;margin-top:2px;">{msg}</div>
        </div>
    </div>
    ''' for icon, color, sev, msg in env_alert_rows)
else:
    env_alert_html = '''
    <div style="text-align:center;margin-top:12px;color:#21e36d;font-size:14px;font-weight:bold;">
        ● EMISSIONS WITHIN SAFE LIMITS
        <br><span style="color:#6a8a9a;font-size:12px;font-weight:normal;">No environmental alerts</span>
    </div>
    '''


# ============================================================
# MAIN DASHBOARD HTML
# ============================================================

dashboard_html = f"""
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<style>

*{{box-sizing:border-box;margin:0;padding:0;}}

html,body{{
    width:100%;
    background:#03070d;
    color:#fff;
    font-family:Arial,Helvetica,sans-serif;
    overflow-x:hidden;
    overflow-y:auto;
}}

.dash{{
    padding:14px 24px;
    background:radial-gradient(ellipse at center,#0e2840 0%,#060f1a 40%,#03070d 80%);
}}

/* ── HEADER ── */
.hdr{{display:flex;justify-content:space-between;align-items:center;height:64px;border-bottom:1px solid rgba(55,150,255,0.2);margin-bottom:12px;}}
.hdr-left{{width:30%;color:#8fa4b8;font-size:12px;letter-spacing:1px;}}
.hdr-connected{{color:#30e46c;font-size:10px;font-weight:bold;margin-top:5px;}}
.hdr-center{{width:40%;text-align:center;}}
.hdr-logo{{font-size:28px;font-weight:900;letter-spacing:6px;}}
.hdr-sub{{font-size:9px;letter-spacing:5px;color:#7a90a4;margin-top:3px;}}
.hdr-right{{width:30%;text-align:right;color:#8fa4b8;font-size:12px;}}

/* ── COCKPIT ── */
.cockpit{{display:grid;grid-template-columns:1fr 1.4fr 1fr;gap:16px;min-height:480px;}}

/* ── GAUGES ── */
.gauge-wrap{{display:flex;flex-direction:column;justify-content:center;align-items:center;gap:10px;}}
.gauge-label{{color:#6a85a0;font-size:10px;letter-spacing:3px;}}
.gauge-ring{{
    position:relative;width:220px;height:220px;border-radius:50%;
    display:flex;justify-content:center;align-items:center;
    background:conic-gradient(from 220deg,#1570cc 0deg,#1ea7ff 120deg,rgba(30,167,255,0.08) 120deg,rgba(30,167,255,0.08) 360deg);
    box-shadow:0 0 30px rgba(0,120,255,0.3);
}}
.gauge-ring::before{{
    content:"";position:absolute;width:178px;height:178px;border-radius:50%;
    background:radial-gradient(circle,#0d1a26 0%,#040a10 70%);
    border:1px solid #173d5e;
}}
.gauge-inner{{position:relative;z-index:2;text-align:center;}}
.gauge-val{{font-size:52px;font-weight:800;letter-spacing:-2px;}}
.gauge-unit{{font-size:12px;color:#6a85a0;letter-spacing:2px;margin-top:2px;}}
.gauge-sub{{font-size:10px;color:#39a6ff;letter-spacing:1px;margin-top:8px;}}

/* ── CAR ── */
.car-area{{position:relative;display:flex;justify-content:center;align-items:center;background:radial-gradient(ellipse at center,rgba(0,100,255,0.1),transparent 70%);}}
.car-wrap{{position:relative;width:100%;height:100%;min-height:480px;}}
.car-svg{{position:absolute;width:240px;height:400px;top:50%;left:50%;transform:translate(-50%,-50%);}}

/* ── TYRE CARDS ── */
.tc{{
    position:absolute;width:150px;
    background:linear-gradient(145deg,rgba(12,22,34,0.97),rgba(4,8,14,0.97));
    border:1px solid rgba(50,120,180,0.4);border-radius:14px;padding:12px;
}}
.tc-name{{font-size:9px;letter-spacing:2px;color:#5a7a90;margin-bottom:6px;}}
.tc-arc{{display:flex;align-items:center;gap:6px;}}
.tc-psi{{font-size:24px;font-weight:800;line-height:1;}}
.tc-psi-lbl{{font-size:9px;color:#5a7a90;letter-spacing:1px;}}
.tc-temp{{font-size:11px;color:#c0d0dc;margin-top:4px;}}
.tc-fl{{top:40px;left:8px;}}
.tc-fr{{top:40px;right:8px;}}
.tc-rl{{bottom:30px;left:8px;}}
.tc-rr{{bottom:30px;right:8px;}}

/* ── STATUS BAR ── */
.status-bar{{margin:10px 0;padding:12px 20px;text-align:center;border-radius:12px;border:1px solid {overall_color};background:rgba(0,0,0,0.3);}}
.status-lbl{{font-size:10px;color:#6a85a0;letter-spacing:2px;}}
.status-val{{font-size:18px;font-weight:800;color:{overall_color};letter-spacing:3px;margin-top:2px;}}
.status-msg{{font-size:10px;color:#8aa0b0;margin-top:3px;}}

/* ── BOTTOM PANELS ── */
.bottom{{display:grid;grid-template-columns:1fr 1.3fr 1fr;gap:14px;margin-top:12px;}}
.panel{{border-radius:16px;padding:18px;min-height:260px;background:linear-gradient(145deg,#0a1520,#040810);border:1px solid rgba(40,100,150,0.4);}}
.panel-title{{color:#6a85a0;font-size:10px;letter-spacing:2px;font-weight:bold;margin-bottom:10px;}}

/* ── HEALTH ── */
.health-wrap{{display:flex;align-items:center;gap:20px;margin-top:12px;}}
.health-circle{{position:relative;width:120px;height:120px;flex-shrink:0;}}
.health-circle svg{{position:absolute;top:0;left:0;}}
.health-num{{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);text-align:center;}}
.health-big{{font-size:34px;font-weight:800;}}
.health-small{{font-size:11px;color:#5a7a90;}}
.health-rows{{flex:1;}}
.health-row{{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;}}
.health-row-lbl{{font-size:11px;font-weight:bold;}}
.health-row-num{{font-size:13px;font-weight:800;}}

/* ── ALERTS ── */
.alert-scroll{{max-height:280px;overflow-y:auto;padding-right:4px;}}
.alert-scroll::-webkit-scrollbar{{width:3px;}}
.alert-scroll::-webkit-scrollbar-thumb{{background:#1a3a5a;border-radius:3px;}}

/* ── RUL ── */
.rul-grid{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:6px;}}
.rul-item{{background:rgba(255,255,255,0.03);border:1px solid rgba(50,120,180,0.25);border-radius:10px;padding:20px 14px;}}
.rul-pos{{font-size:10px;letter-spacing:2px;color:#5a7a90;margin-bottom:6px;}}
.rul-val{{font-size:26px;font-weight:800;line-height:1;}}
.rul-km{{font-size:11px;color:#5a7a90;margin-left:2px;}}
.rul-bar{{width:100%;height:6px;border-radius:10px;margin-top:6px;background:linear-gradient(90deg,#ff3b3b 0%,#ff9f1c 30%,#ffd23f 55%,#21e36d 80%);position:relative;overflow:visible;}}
.rul-marker{{position:absolute;top:-3px;width:3px;height:12px;background:#fff;border-radius:2px;}}

/* ── FOOTER ── */
.footer{{text-align:center;color:#3a5060;font-size:16px;letter-spacing:3px;padding:16px 0 8px 0;}}

/* ── ENV ── */
.env-panel{{border-radius:16px;padding:16px 20px;background:linear-gradient(145deg,#0a1a10,#040810);border:1px solid rgba(30,180,80,0.3);margin-top:12px;}}
.env-grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-top:10px;}}
.env-card{{background:rgba(255,255,255,0.03);border:1px solid rgba(50,180,100,0.2);border-radius:10px;padding:10px;text-align:center;}}
.env-label{{font-size:8px;letter-spacing:1.5px;color:#5a8a6a;margin-bottom:6px;}}
.env-val{{font-size:18px;font-weight:800;line-height:1;}}
.env-unit{{font-size:8px;color:#4a6a5a;margin-top:3px;}}

/* ── MOBILE ── */
@media screen and (max-width:680px){{
    .dash{{padding:10px 8px;}}

    .hdr{{height:auto;flex-direction:column;text-align:center;gap:4px;padding-bottom:10px;}}
    .hdr-left{{width:100%;text-align:center;}}
    .hdr-center{{width:100%;text-align:center;}}
    .hdr-right{{width:100%;text-align:center;}}

    .cockpit{{grid-template-columns:1fr;min-height:auto;gap:8px;}}
    .gauge-wrap{{display:none;}}
    .car-area{{min-height:340px;}}
    .car-wrap{{min-height:340px;}}
    .car-svg{{width:150px;height:260px;}}

    .tc{{width:108px;padding:8px;}}
    .tc-psi{{font-size:17px;}}
    .tc-temp{{font-size:9px;}}
    .tc-fl{{top:18px;left:2px;}}
    .tc-fr{{top:18px;right:2px;}}
    .tc-rl{{bottom:10px;left:2px;}}
    .tc-rr{{bottom:10px;right:2px;}}

    .status-val{{font-size:14px;}}

    .bottom{{grid-template-columns:1fr;gap:10px;}}
    .panel{{min-height:auto;}}

    .health-wrap{{flex-direction:column;align-items:center;}}

    .rul-val{{font-size:20px;}}
    .rul-item{{padding:12px 10px;}}

    .footer{{font-size:11px;letter-spacing:1px;}}
}}

</style>


</head>


<body>

<div class="dash">


<!-- ===================================================== -->
<!-- HEADER -->
<!-- ===================================================== -->

<div class="hdr">

    <div class="hdr-left">

        🕒 {current_time}

        <div class="hdr-connected">
            ⬤ VEHICLE CONNECTED
        </div>

    </div>


    <div class="hdr-center">

        <div class="hdr-logo">

            🛞 T
            <span style="color:#32a0ff;">
                P
            </span>
            A

        </div>


        <div class="hdr-sub">
            TYRE PERSONAL ASSISTANT
        </div>

    </div>


    <div class="hdr-right">

        LIVE TELEMETRY

        <br>

        <span style="
            color:#39a6ff;
            font-size:10px;
        ">

            +{simulation_minutes} MIN

        </span>

        <br>

        <span style="
            color:#4a7a9b;
            font-size:9px;
            letter-spacing:1px;
        ">
            📍 ~{_sim_dist:,.0f} km / step &nbsp;·&nbsp; {_total_dist:,.0f} km driven
        </span>

    </div>

</div>


<!-- ===================================================== -->
<!-- COCKPIT -->
<!-- ===================================================== -->

<div class="cockpit">


<!-- SPEED -->

<div class="gauge-wrap">

    <div class="gauge-label">
        VEHICLE SPEED
    </div>


    <div class="gauge-ring">

        <div class="gauge-inner">

            <div class="gauge-val">
                {speed:.0f}
            </div>


            <div class="gauge-unit">
                KM/H
            </div>


            <div class="gauge-sub">
                LIVE SPEED
            </div>

        </div>

    </div>

</div>


<!-- CAR -->

<div class="car-area">

<div class="car-wrap">


<!-- FRONT LEFT -->

<div class="tc tc-fl">

    <div class="tc-name">
        FRONT LEFT
    </div>


    <div class="tc-arc">

        {tyre_arc(
            fl_data["pressure"],
            40,
            fl_data["color"],
            70
        )}


        <div>

            <div
                class="tc-psi"
                style="
                    color:
                    {fl_data['color']};
                "
            >

                {fl_data["pressure"]:.1f}

            </div>


            <div class="tc-psi-lbl">
                PSI
            </div>


            <div class="tc-temp">

                🌡
                {fl_data["temperature"]:.1f}°C

            </div>

        </div>

    </div>


    {status_badge(
        fl_data["status"],
        fl_data["color"]
    )}

</div>


<!-- FRONT RIGHT -->

<div class="tc tc-fr">

    <div class="tc-name">
        FRONT RIGHT
    </div>


    <div class="tc-arc">

        {tyre_arc(
            fr_data["pressure"],
            40,
            fr_data["color"],
            70
        )}


        <div>

            <div
                class="tc-psi"
                style="
                    color:
                    {fr_data['color']};
                "
            >

                {fr_data["pressure"]:.1f}

            </div>


            <div class="tc-psi-lbl">
                PSI
            </div>


            <div class="tc-temp">

                🌡
                {fr_data["temperature"]:.1f}°C

            </div>

        </div>

    </div>


    {status_badge(
        fr_data["status"],
        fr_data["color"]
    )}

</div>


<!-- REAR LEFT -->

<div class="tc tc-rl">

    <div class="tc-name">
        REAR LEFT
    </div>


    <div class="tc-arc">

        {tyre_arc(
            rl_data["pressure"],
            40,
            rl_data["color"],
            70
        )}


        <div>

            <div
                class="tc-psi"
                style="
                    color:
                    {rl_data['color']};
                "
            >

                {rl_data["pressure"]:.1f}

            </div>


            <div class="tc-psi-lbl">
                PSI
            </div>


            <div class="tc-temp">

                🌡
                {rl_data["temperature"]:.1f}°C

            </div>

        </div>

    </div>


    {status_badge(
        rl_data["status"],
        rl_data["color"]
    )}

</div>


<!-- REAR RIGHT -->

<div class="tc tc-rr">

    <div class="tc-name">
        REAR RIGHT
    </div>


    <div class="tc-arc">

        {tyre_arc(
            rr_data["pressure"],
            40,
            rr_data["color"],
            70
        )}


        <div>

            <div
                class="tc-psi"
                style="
                    color:
                    {rr_data['color']};
                "
            >

                {rr_data["pressure"]:.1f}

            </div>


            <div class="tc-psi-lbl">
                PSI
            </div>


            <div class="tc-temp">

                🌡
                {rr_data["temperature"]:.1f}°C

            </div>

        </div>

    </div>


    {status_badge(
        rr_data["status"],
        rr_data["color"]
    )}

</div>


<!-- CAR SVG -->

<svg
    class="car-svg"
    viewBox="0 0 260 440"
    xmlns="http://www.w3.org/2000/svg"
>

<defs>

    <filter id="glow">

        <feGaussianBlur
            stdDeviation="4"
            result="blur"
        />

        <feMerge>

            <feMergeNode
                in="blur"
            />

            <feMergeNode
                in="SourceGraphic"
            />

        </feMerge>

    </filter>


    <linearGradient
        id="carBody"
        x1="0"
        y1="0"
        x2="0"
        y2="1"
    >

        <stop
            offset="0%"
            stop-color="#2a4060"
        />

        <stop
            offset="50%"
            stop-color="#0d1a28"
        />

        <stop
            offset="100%"
            stop-color="#040a12"
        />

    </linearGradient>


    <linearGradient
        id="glass"
        x1="0"
        y1="0"
        x2="0"
        y2="1"
    >

        <stop
            offset="0%"
            stop-color="#1e3a52"
        />

        <stop
            offset="100%"
            stop-color="#060e16"
        />

    </linearGradient>

</defs>


<path
    d="
        M88 25
        C68 45
        60 90
        55 145
        L42 285
        C40 340
        63 405
        94 425
        L166 425
        C197 405
        220 340
        218 285
        L205 145
        C200 90
        192 45
        172 25
        Q130 5
        88 25
    "

    fill="url(#carBody)"

    stroke="#3a90e0"

    stroke-width="1.5"

    filter="url(#glow)"
/>


<path
    d="
        M88 65
        L172 65
        L192 175
        L68 175
        Z
    "

    fill="url(#glass)"

    stroke="#5aaeff"

    stroke-width="1.2"
/>


<path
    d="
        M70 275
        L190 275
        L178 355
        L82 355
        Z
    "

    fill="url(#glass)"

    stroke="#3a6080"

    stroke-width="1"
/>


<line
    x1="130"
    y1="30"
    x2="130"
    y2="410"

    stroke="#2a5070"

    stroke-width="1"

    stroke-dasharray="5 8"
/>


<path
    d="M55 105 L85 135"

    stroke="#38b6ff"

    stroke-width="6"

    stroke-linecap="round"

    filter="url(#glow)"
/>


<path
    d="M205 105 L175 135"

    stroke="#38b6ff"

    stroke-width="6"

    stroke-linecap="round"

    filter="url(#glow)"
/>


<path
    d="M52 370 L84 340"

    stroke="#38b6ff"

    stroke-width="6"

    stroke-linecap="round"
    filter="url(#glow)"
/>


<path
    d="M208 370 L176 340"

    stroke="#38b6ff"

    stroke-width="6"

    stroke-linecap="round"
    filter="url(#glow)"
/>


<text
    x="130"
    y="222"

    text-anchor="middle"

    fill="#4aa0f0"

    font-size="20"

    font-weight="bold"

    font-family="Arial"
>

    TPA

</text>


<text
    x="130"
    y="240"

    text-anchor="middle"

    fill="#5a7a90"

    font-size="7"

    font-family="Arial"
>

    TYRE INTELLIGENCE

</text>


</svg>


</div>
</div>


<!-- RPM -->

<div class="gauge-wrap">

    <div class="gauge-label">
        ENGINE RPM
    </div>


    <div class="gauge-ring">

        <div class="gauge-inner">

            <div class="gauge-val">
                {rpm:.1f}
            </div>


            <div class="gauge-unit">
                x1000 RPM
            </div>


            <div class="gauge-sub">
                AVG
                {avg_temp:.1f}°C
            </div>

        </div>

    </div>

</div>


</div>


<!-- ===================================================== -->
<!-- OVERALL STATUS + ENV -->
<div class="status-bar">
    <div class="status-lbl">OVERALL STATUS</div>
    <div class="status-val">⚠ {overall_status}</div>
    <div class="status-msg">{overall_message}</div>
</div>


<!-- ===================================================== -->
<!-- BOTTOM -->
<!-- ===================================================== -->

<div class="bottom">


<!-- HEALTH SCORE -->

<div class="panel">

    <div class="panel-title">
        TYRE HEALTH SCORE
    </div>


    <div class="health-wrap">


        <div class="health-circle">

            <svg
                width="120"
                height="120"
                viewBox="0 0 100 100"
            >

                <circle
                    cx="50"
                    cy="50"
                    r="40"

                    fill="none"

                    stroke="#0d1e2e"

                    stroke-width="10"
                />


                <circle
                    cx="50"
                    cy="50"
                    r="40"

                    fill="none"

                    stroke="#21e36d"

                    stroke-width="10"

                    stroke-dasharray="
                        {overall_health * 2.513:.1f}
                        251.3
                    "

                    stroke-linecap="round"

                    transform="
                        rotate(-90 50 50)
                    "
                />

            </svg>


            <div class="health-num">

                <div class="health-big">
                    {overall_health:.0f}
                </div>


                <div class="health-small">
                    /100
                </div>

            </div>

        </div>


        <div class="health-rows">


            <div class="health-row">

                <span
                    class="health-row-lbl"
                    style="color:#21e36d;"
                >

                    HEALTHY

                </span>


                <span class="health-row-num">

                    {status_counts["HEALTHY"]}

                </span>

            </div>


            <div class="health-row">

                <span
                    class="health-row-lbl"
                    style="color:#ffd23f;"
                >

                    ATTENTION

                </span>


                <span class="health-row-num">

                    {status_counts["ATTENTION"]}

                </span>

            </div>


            <div class="health-row">

                <span
                    class="health-row-lbl"
                    style="color:#ff9f1c;"
                >

                    WARNING

                </span>


                <span class="health-row-num">

                    {status_counts["WARNING"]}

                </span>

            </div>


            <div class="health-row">

                <span
                    class="health-row-lbl"
                    style="color:#ff3b3b;"
                >

                    CRITICAL

                </span>


                <span class="health-row-num">

                    {status_counts["CRITICAL"]}

                </span>

            </div>


        </div>


    </div>

    <div style="border-top:1px solid rgba(30,180,80,0.2);margin-top:12px;padding-top:10px;">
        <div style="color:#3ab870;font-size:13px;letter-spacing:2px;font-weight:bold;margin-bottom:6px;">🌿 ENVIRONMENTAL ALERTS</div>
        {env_alert_html}
    </div>

</div>


<!-- TPA ALERTS -->

<div class="panel">

    <div
        class="panel-title"
        style="
            text-align:center;
            font-size:12px;
        "
    >

        🚨 TPA ALERTS

    </div>


    <div class="alert-scroll">

        {alert_html}

    </div>

</div>


<!-- REMAINING LIFE -->

<div class="panel">

    <div class="panel-title">

        PREDICTED REMAINING LIFE

    </div>


    <div class="rul-grid">

        {rul_cards_html}

    </div>

</div>


</div>


<div class="footer">

    TPA
    •
    TYRE PERSONAL ASSISTANT
    •
    AI
    •
    IOT
    •
    PREDICTIVE MAINTENANCE

</div>


</div>


</body>

</html>
"""


# ============================================================
# RENDER DASHBOARD
# ============================================================

components.html(
    dashboard_html,
    height=1800,
    scrolling=True
)


# ============================================================
# MODEL INFO SECTION
# ============================================================

st.markdown(
    "<hr style='border:1px solid rgba(50,120,180,0.3);margin:8px 0 16px 0;'>",
    unsafe_allow_html=True
)

with st.expander("🤖  MODEL INTELLIGENCE OVERVIEW", expanded=False):

    st.markdown(
        "<div style='text-align:center;color:#39a6ff;font-size:22px;"
        "letter-spacing:4px;font-weight:900;margin-bottom:18px;'>"
        "🤖 MODEL INTELLIGENCE OVERVIEW</div>",
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    # ── Health Classification Model ──────────────────────────────
    with col1:
        with st.expander("🛞 Health Classification Model", expanded=True):
            st.markdown("**Input Features**")
            health_features = predictor.features
            for i, f in enumerate(health_features, 1):
                st.markdown(f"`{i:02d}` {f}")
            st.divider()
            st.markdown("**Predicted Output**")
            st.markdown("`health_status` — classification label")
            classes = predictor.label_encoder.classes_
            st.markdown(
                " &nbsp;".join(
                    [f"`{c}`" for c in classes]
                ),
                unsafe_allow_html=True
            )

    # ── RUL Regression Model ─────────────────────────────────────
    with col2:
        with st.expander("📏 Remaining Useful Life Model", expanded=True):
            st.markdown("**Input Features**")
            rul_features = predictor.features  # same feature set
            for i, f in enumerate(rul_features, 1):
                st.markdown(f"`{i:02d}` {f}")
            st.divider()
            st.markdown("**Predicted Output**")
            st.markdown("`remaining_life_km` — continuous value (km)")

    # ── Environmental Impact Model ────────────────────────────────
    with col3:
        with st.expander("🌍 Environmental Impact Model", expanded=True):
            st.markdown("**Input Features**")
            for i, f in enumerate(env_predictor.features, 1):
                st.markdown(f"`{i:02d}` {f}")
            st.divider()
            st.markdown("**Predicted Outputs**")
            env_output_desc = {
                "abrasion_rate_mg_km":   "Tyre abrasion rate (mg/km)",
                "microplastic_g_per_km": "Microplastic shedding (g/km)",
            }
            for col_name, desc in env_output_desc.items():
                st.markdown(f"`{col_name}` — {desc}")


# ============================================================
# DATA SAMPLE SECTION
# ============================================================

st.markdown(
    "<hr style='border:1px solid rgba(50,120,180,0.3);margin:16px 0;'>",
    unsafe_allow_html=True
)

with st.expander("📊  DATA SAMPLE", expanded=False):

    st.markdown(
        "<div style='text-align:center;color:#39a6ff;font-size:22px;"
        "letter-spacing:4px;font-weight:900;margin-bottom:18px;'>"
        "📊 DATA SAMPLE</div>",
        unsafe_allow_html=True
    )

    # ── Tabs: Live Predictions  |  Historical Dataset ────────────
    tab_live, tab_hist = st.tabs([
        "⚡ Live Prediction Data",
        "🗄️ Historical Training Data"
    ])

    # Column groups for selective display
    SENSOR_COLS = [
        "tyre_position", "tyre_age_days", "distance_km",
        "pressure_psi", "temperature_c", "vibration",
        "speed_kmh", "vehicle_load_kg", "braking_intensity",
        "pressure_change", "temperature_change", "wear_ratio",
    ]
    PREDICTION_COLS = [
        "tyre_position", "health_status", "confidence",
        "remaining_life_km",
        "abrasion_rate_mg_km", "microplastic_g_per_km",
    ]
    HIST_FEATURE_COLS = [
        "timestamp", "tyre_position", "tyre_age_days", "distance_km",
        "pressure_psi", "temperature_c", "vibration", "speed_kmh",
        "road_condition", "weather_condition", "vehicle_load_kg",
        "braking_intensity", "wear_ratio",
    ]
    HIST_LABEL_COLS = [
        "timestamp", "tyre_position",
        "tyre_health_status", "remaining_life_km",
        "abrasion_rate_mg_km", "microplastic_g_per_km",
    ]

    # ── Tab 1: Live Prediction Data ───────────────────────────────
    with tab_live:

        view = st.radio(
            "Show columns",
            ["Sensor Inputs", "Model Predictions", "All Columns"],
            horizontal=True,
            key="live_view"
        )

        live_df = predictions.copy()

        if view == "Sensor Inputs":
            show_cols = [c for c in SENSOR_COLS if c in live_df.columns]
        elif view == "Model Predictions":
            show_cols = [c for c in PREDICTION_COLS if c in live_df.columns]
        else:
            show_cols = list(live_df.columns)

        display_live = live_df[show_cols].reset_index(drop=True)

        # Colour-code health_status if present
        def highlight_health(val):
            colors = {
                "HEALTHY":   "background-color:#0d3d1a;color:#21e36d",
                "ATTENTION": "background-color:#3d3200;color:#ffd23f",
                "WARNING":   "background-color:#3d1e00;color:#ff9f1c",
                "CRITICAL":  "background-color:#3d0000;color:#ff3b3b",
            }
            return colors.get(str(val).upper(), "")

        if "health_status" in display_live.columns:
            st.dataframe(
                display_live.style.applymap(
                    highlight_health,
                    subset=["health_status"]
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.dataframe(display_live, use_container_width=True, hide_index=True)

        st.caption(f"4 live tyres · {len(show_cols)} columns shown · updates on each simulation step")

    # ── Tab 2: Historical Training Data ──────────────────────────
    with tab_hist:

        @st.cache_data
        def load_historical():
            import os
            path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "data", "tyre_historical_data.csv"
            )
            return pd.read_csv(path)

        hist_df = load_historical()

        hcol1, hcol2, hcol3 = st.columns([1, 1, 1])

        with hcol1:
            hview = st.radio(
                "Show columns",
                ["Feature Inputs", "Target Labels", "All Columns"],
                horizontal=False,
                key="hist_view"
            )
        with hcol2:
            health_filter = st.multiselect(
                "Filter by health status",
                options=["HEALTHY", "ATTENTION", "WARNING", "CRITICAL"],
                default=[],
                key="hist_health_filter"
            )
        with hcol3:
            n_rows = st.slider(
                "Rows to preview",
                min_value=5, max_value=100,
                value=10, step=5,
                key="hist_rows"
            )

        # Apply health filter
        filtered_hist = hist_df.copy()
        if health_filter:
            filtered_hist = filtered_hist[
                filtered_hist["tyre_health_status"].isin(health_filter)
            ]

        # Select column view
        if hview == "Feature Inputs":
            hcols = [c for c in HIST_FEATURE_COLS if c in filtered_hist.columns]
        elif hview == "Target Labels":
            hcols = [c for c in HIST_LABEL_COLS if c in filtered_hist.columns]
        else:
            hcols = list(filtered_hist.columns)

        display_hist = filtered_hist[hcols].head(n_rows).reset_index(drop=True)

        if "tyre_health_status" in display_hist.columns:
            def highlight_hist_health(val):
                colors = {
                    "HEALTHY":   "background-color:#0d3d1a;color:#21e36d",
                    "ATTENTION": "background-color:#3d3200;color:#ffd23f",
                    "WARNING":   "background-color:#3d1e00;color:#ff9f1c",
                    "CRITICAL":  "background-color:#3d0000;color:#ff3b3b",
                }
                return colors.get(str(val).upper(), "")

            st.dataframe(
                display_hist.style.applymap(
                    highlight_hist_health,
                    subset=["tyre_health_status"]
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.dataframe(display_hist, use_container_width=True, hide_index=True)

        st.caption(
            f"{len(filtered_hist):,} rows after filter · "
            f"{len(hcols)} columns shown · "
            f"total dataset: {len(hist_df):,} rows × {len(hist_df.columns)} columns"
        )


# ============================================================
# DATA GENERATION FORMULA GUIDE
# ============================================================

st.markdown(
    "<hr style='border:1px solid rgba(50,120,180,0.3);margin:16px 0;'>",
    unsafe_allow_html=True
)

with st.expander("⚙️  DATA GENERATION — FORMULAS & CALCULATIONS", expanded=False):

    st.markdown(
        "<div style='text-align:center;color:#39a6ff;font-size:22px;"
        "letter-spacing:4px;font-weight:900;margin-bottom:4px;'>"
        "⚙️ DATA GENERATION — FORMULAS &amp; CALCULATIONS</div>"
        "<div style='text-align:center;color:#6a85a0;font-size:12px;"
        "letter-spacing:2px;margin-bottom:20px;'>"
        "How every field in the dataset is computed — all factors explained</div>",
        unsafe_allow_html=True
    )

    # ── DATASET SCALE ────────────────────────────────────────
    st.markdown("### 📦 Dataset Scale")
    scale_col1, scale_col2, scale_col3, scale_col4 = st.columns(4)
    scale_col1.metric("Vehicles", "100")
    scale_col2.metric("Tyres per Vehicle", "4  (FL · FR · RL · RR)")
    scale_col3.metric("Readings per Tyre", "250  (every 10 min)")
    scale_col4.metric("Total Records", "100,000")

    st.divider()

    # ── TYRE SCENARIOS ───────────────────────────────────────
    st.markdown("### 🎭 Tyre Scenario Assignment")
    st.markdown(
        "Each tyre is randomly assigned one degradation scenario at birth. "
        "This scenario controls how pressure, temperature, and vibration "
        "evolve across the 250 readings."
    )
    scenario_data = {
        "Scenario": ["healthy", "slow_leak", "overheating", "high_vibration", "critical"],
        "Probability": ["50%", "20%", "12%", "10%", "8%"],
        "Pressure Behaviour": [
            "Stable ±0.25 PSI drift, clamped [32–36]",
            "Drops −0.02 to −0.08 PSI per reading (slow deflation)",
            "Drops −0.00 to −0.03 PSI per reading",
            "Random ±0.30 drift (no directional trend)",
            "Drops −0.05 to −0.15 PSI per reading (fast leak)"
        ],
        "Temperature Behaviour": [
            "25 + speed×0.25 + noise(0,2) — normal heating",
            "28 + speed×0.30 + noise(0,3) — slightly elevated",
            "Cumulative rise: +0.05 to +0.30 + speed×0.05 per step",
            "30 + speed×0.30 + noise(0,4)",
            "Cumulative rise: +0.10 to +0.50 per step (runaway)"
        ],
        "Vibration Behaviour": [
            "Normal(0.12, 0.03) — stable low vibration",
            "Normal(0.18, 0.05) — slightly higher",
            "Normal(0.20, 0.06) — elevated",
            "Increments +0.01 to +0.05 per reading (increasing)",
            "Increments +0.02 to +0.08 per reading (severe)"
        ],
    }
    st.dataframe(scenario_data, use_container_width=True, hide_index=True)

    st.divider()

    # ── SENSOR FIELDS ────────────────────────────────────────
    st.markdown("### 📡 Sensor & Driving Fields — How Each Is Generated")

    fc1, fc2 = st.columns(2)

    with fc1:
        st.markdown("#### 🚗 Speed  `speed_kmh`")
        st.code("speed_kmh = clip( Normal(mean=65, std=20), 0, 140 )", language="python")
        st.caption("Normal distribution centred at 65 km/h, std=20. Clipped to [0, 140].")

        st.markdown("#### 🌦️ Weather  `weather_condition`")
        st.code(
            "weather = choice(['DRY','WET','RAIN','SNOW'],\n"
            "                  weights=[0.55, 0.25, 0.15, 0.05])",
            language="python"
        )
        st.caption("DRY = 55% of readings. SNOW = 5% (rare).")

        st.markdown("#### 🛣️ Road Condition  `road_condition`")
        st.code(
            "road = choice(['GOOD','NORMAL','ROUGH'],\n"
            "               weights=[0.45, 0.40, 0.15])",
            language="python"
        )
        st.caption("GOOD = 45%, ROUGH = 15% of readings.")

        st.markdown("#### ⚖️ Vehicle Load  `vehicle_load_kg`")
        st.code(
            "base_load ~ Uniform(300, 600) per vehicle\n"
            "load = clip( base_load + Normal(0, 50), 200, 800 )",
            language="python"
        )
        st.caption("Each vehicle has a base load. Per reading, load fluctuates ±50 kg (passengers/cargo).")

    with fc2:
        st.markdown("#### 🛑 Braking Intensity  `braking_intensity`")
        st.code(
            "brake_base = 0.05\n"
            "if speed_kmh > 80:     brake_base += 0.10\n"
            "if road == 'ROUGH':    brake_base += 0.10\n"
            "if weather in RAIN/SNOW: brake_base += 0.15\n\n"
            "braking = clip( Exponential(brake_base), 0.0, 1.0 )",
            language="python"
        )
        st.caption(
            "Exponential distribution (rare hard braking, frequent gentle). "
            "High speed, rough roads, and bad weather increase the rate."
        )

        st.markdown("#### 📏 Distance Increment  `distance_km`")
        st.code(
            "distance_increment = speed_kmh / 6\n"
            "# speed_kmh / 6 = km driven in 10 minutes\n"
            "distance_km += distance_increment",
            language="python"
        )
        st.caption("At 60 km/h → +10 km per reading. At 120 km/h → +20 km per reading.")

        st.markdown("#### 🔄 Pressure & Temperature Change")
        st.code(
            "pressure_change    = new_pressure - prev_pressure\n"
            "temperature_change = new_temperature - prev_temperature",
            language="python"
        )
        st.caption("Simple delta between consecutive readings. Negative = drop, Positive = rise.")

    st.divider()

    # ── WEATHER & LOAD EFFECTS ────────────────────────────────
    st.markdown("### 🌡️ Weather & Load Corrections Applied to Pressure & Temperature")

    wl_col1, wl_col2 = st.columns(2)
    with wl_col1:
        st.markdown("**Weather → Pressure correction**")
        st.code(
            "if weather == 'SNOW':\n"
            "    pressure -= Uniform(0.1, 0.3)  # cold contracts air\n"
            "# RAIN, DRY, WET: no pressure correction",
            language="python"
        )
        st.markdown("**Weather → Temperature correction**")
        st.code(
            "if weather == 'SNOW': temperature -= Uniform(2, 5)\n"
            "if weather == 'RAIN': temperature -= Uniform(1, 3)",
            language="python"
        )

    with wl_col2:
        st.markdown("**Load → Temperature correction**")
        st.code(
            "temperature += (vehicle_load_kg - 400) / 400 * 3.0\n"
            "# +400 kg above reference → +3°C\n"
            "# −200 kg below reference → −1.5°C",
            language="python"
        )
        st.markdown("**Braking → Temperature correction**")
        st.code(
            "temperature += braking_intensity * 8.0\n"
            "# emergency brake (1.0) → +8°C\n"
            "# gentle brake (0.05) → +0.4°C",
            language="python"
        )

    st.markdown("**Road → Vibration correction**")
    st.code(
        "if road_condition == 'ROUGH':\n"
        "    vibration += Uniform(0.03, 0.10)  # rough road adds vibration",
        language="python"
    )
    st.caption("All sensor values are clamped after corrections: pressure [10–40], temperature [15–120], vibration [0.01–1.50]")

    st.divider()

    # ── WEAR FACTOR ──────────────────────────────────────────
    st.markdown("### 🔧 Wear Factor — `wear_factor` (Accelerated Degradation Multiplier)")
    st.markdown(
        "A multiplier applied to actual distance to compute **effective usage**. "
        "Poor conditions make the tyre wear faster than the raw km suggest."
    )
    st.code(
        "wear_factor = 1.0  # baseline\n\n"
        "if pressure < 32:          wear_factor += (32 - pressure) * 0.04\n"
        "# Under-inflation → higher flex → faster wear\n\n"
        "if temperature > 55:       wear_factor += (temperature - 55) * 0.01\n"
        "# Overheating softens rubber → faster wear\n\n"
        "if vibration > 0.30:       wear_factor += (vibration - 0.30) * 0.50\n"
        "# Imbalance/rough road → uneven rapid wear\n\n"
        "if speed_kmh > 100:        wear_factor += (speed_kmh - 100) * 0.005\n"
        "# High speed increases centrifugal stress\n\n"
        "if road == 'ROUGH':        wear_factor += 0.05\n"
        "# Rough surface = constant +5% wear\n\n"
        "if tyre_age_days > 1095:   wear_factor += 0.15\n"
        "# Old tyres (>3 years) degrade faster regardless of condition\n\n"
        "if braking_intensity > 0.5: wear_factor += braking_intensity * 0.20\n"
        "# Hard braking scrubs rubber\n\n"
        "if weather == 'SNOW':      wear_factor += 0.10\n"
        "# Snow/ice increases grip load on compound",
        language="python"
    )

    st.divider()

    # ── REMAINING LIFE ───────────────────────────────────────
    st.markdown("### 📉 Remaining Useful Life  `remaining_life_km`")
    st.code(
        "effective_usage   = distance_km * wear_factor\n"
        "remaining_life_km = max(0, BASE_TYRE_LIFE_KM - effective_usage)\n"
        "remaining_life_km += Normal(0, 500)   # real-world variability\n"
        "remaining_life_km  = clip(remaining_life_km, 0, 60_000)\n\n"
        "BASE_TYRE_LIFE_KM = 60,000 km",
        language="python"
    )
    st.caption(
        "Effective usage = raw km × wear multiplier. A tyre driven 30,000 km "
        "under-inflated at high speed could have an effective usage of 40,000+ km, "
        "meaning less remaining life than distance alone would suggest."
    )

    st.divider()

    # ── WEAR RATIO ───────────────────────────────────────────
    st.markdown("### ⚙️ Wear Ratio  `wear_ratio`")
    st.code(
        "wear_ratio = min(1.0, distance_km / 60_000)\n"
        "# 0.0 = brand new   →   1.0 = fully worn",
        language="python"
    )
    st.caption("Simple fraction of tyre life consumed by distance. Does NOT include the wear_factor multiplier — it is the raw distance fraction.")

    st.divider()

    # ── HEALTH STATUS / RISK SCORE ───────────────────────────
    st.markdown("### 🏥 Health Status Label  `tyre_health_status`")
    st.markdown(
        "A `risk_score` is computed from physics violations and the scenario. "
        "It is the **ground truth label** the ML model learns to predict."
    )
    st.code(
        "risk_score = 0\n\n"
        "if pressure < 32:            risk_score += (32 - pressure) * 8\n"
        "# Every 1 PSI under 32 adds 8 points\n\n"
        "if temperature > 55:         risk_score += (temperature - 55) * 1.2\n"
        "# Every 1°C over 55 adds 1.2 points\n\n"
        "if vibration > 0.30:         risk_score += (vibration - 0.30) * 30\n"
        "# Every 0.1 above 0.30 adds 3 points\n\n"
        "if remaining_life_km < 5000: risk_score += 30\n"
        "# Near end-of-life is always high risk\n\n"
        "if scenario == 'critical':   risk_score += 20\n"
        "# Critical scenario baseline penalty",
        language="python"
    )

    rs_col1, rs_col2, rs_col3, rs_col4 = st.columns(4)
    rs_col1.metric("HEALTHY",   "risk_score < 10",  "~50% of records")
    rs_col2.metric("ATTENTION", "10 ≤ score < 25",  "~25% of records")
    rs_col3.metric("WARNING",   "25 ≤ score < 50",  "~15% of records")
    rs_col4.metric("CRITICAL",  "score ≥ 50",       "~10% of records")

    st.divider()

    # ── ENVIRONMENTAL METRICS ────────────────────────────────
    st.markdown("### 🌍 Environmental Metrics — Full Formula")
    st.markdown(
        "All three environmental outputs derive from a single physics formula "
        "calibrated to EU tyre wear research (baseline ~80 mg/km at 60 km/h, dry, 400 kg load)."
    )

    env_col1, env_col2 = st.columns(2)

    with env_col1:
        st.markdown("**Step 1 — Compute each factor**")
        st.code(
            "base = 80.0  # mg/km reference\n\n"
            "speed_factor  = (speed_kmh / 60.0) ** 1.5\n"
            "# 30km/h→0.354  60km/h→1.0  120km/h→2.828\n\n"
            "load_factor   = 1.0 + max(0,(load_kg-400)/400)*0.5\n"
            "# 400kg→1.0  600kg→1.25  800kg→1.50\n\n"
            "brake_factor  = 1.0 + braking_intensity * 2.5\n"
            "# 0.0→1.0  0.5→2.25  1.0→3.5\n\n"
            "wear_factor   = 1.0 + wear_ratio * 1.2\n"
            "# new(0.0)→1.0  50%(0.5)→1.6  worn(1.0)→2.2\n\n"
            "vib_factor    = 1.0 + max(0,vibration-0.10)*1.5\n"
            "# 0.10→1.0  0.50→1.6  1.00→2.35",
            language="python"
        )

    with env_col2:
        st.markdown("**Step 2 — Weather multiplier**")
        st.code(
            "weather_factor = {\n"
            "    'DRY' : 1.00,  # baseline\n"
            "    'WET' : 0.85,  # water lubricates\n"
            "    'RAIN': 0.80,  # more lubrication\n"
            "    'SNOW': 1.30,  # chains/grip\n"
            "}",
            language="python"
        )
        st.markdown("**Step 3 — Final abrasion rate**")
        st.code(
            "abrasion_rate = (\n"
            "    base\n"
            "    * speed_factor\n"
            "    * load_factor\n"
            "    * brake_factor\n"
            "    * weather_factor\n"
            "    * wear_factor\n"
            "    * vib_factor\n"
            ")  # unit: mg/km",
            language="python"
        )
        st.markdown("**Step 4 — Derive pollution targets**")
        st.code(
            "microplastic_g_per_km = abrasion * 0.60 / 1000\n"
            "# 60% of worn rubber → particles <5mm",
            language="python"
        )

    st.divider()

    # ── FACTOR IMPACT TABLE ──────────────────────────────────
    st.markdown("### 📊 Factor Impact Reference Table")
    impact_data = {
        "Factor": [
            "Speed 30 km/h", "Speed 60 km/h", "Speed 120 km/h",
            "Load 200 kg", "Load 400 kg (ref)", "Load 800 kg",
            "Braking 0.0 (none)", "Braking 0.5 (moderate)", "Braking 1.0 (emergency)",
            "DRY road", "WET road", "RAIN", "SNOW",
            "New tyre (wear=0.0)", "Half-worn (wear=0.5)", "Fully worn (wear=1.0)",
            "Vibration 0.10 (baseline)", "Vibration 0.50", "Vibration 1.00"
        ],
        "Multiplier Applied": [
            "× 0.354", "× 1.000", "× 2.828",
            "× 1.000", "× 1.000", "× 1.500",
            "× 1.000", "× 2.250", "× 3.500",
            "× 1.000", "× 0.850", "× 0.800", "× 1.300",
            "× 1.000", "× 1.600", "× 2.200",
            "× 1.000", "× 1.600", "× 2.350"
        ],
        "Effect on Abrasion": [
            "65% less than ref", "Baseline reference", "183% more than ref",
            "No extra", "Baseline reference", "50% more",
            "No extra", "125% more", "250% more",
            "Baseline", "15% less", "20% less", "30% more",
            "Baseline", "60% more", "120% more",
            "Baseline", "60% more", "135% more"
        ]
    }
    st.dataframe(impact_data, use_container_width=True, hide_index=True)

    st.caption(
        "All factors multiply together. Example: 120 km/h + emergency braking + "
        "fully worn + SNOW = 2.828 × 3.5 × 2.2 × 1.3 = ~28× baseline abrasion (2,240 mg/km)"
    )


# ============================================================
