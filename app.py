"""
app.py  —  Patient Attendance Intelligence Console
Premium medical SaaS UI redesign. No emojis. Professional. Action-focused.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
from datetime import datetime
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Patient Attendance Intelligence Console",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    box-sizing: border-box;
}

/* ── Reset Streamlit chrome ────────────────────────────────────── */
#MainMenu, header, footer { visibility: hidden; }
section[data-testid="stSidebar"] { display: none !important; }
.block-container {
    padding: 0 3rem 3rem 3rem !important;
    max-width: 100% !important;
}
div[data-testid="stAppViewContainer"] { background: #d4dbe6; }

/* ── TOP BAR ───────────────────────────────────────────────────── */
.top-bar {
    background: #0d1b2a;
    color: white;
    padding: 16px 40px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    margin-left: -3rem;
    margin-right: -3rem;
}
.top-bar-left .app-name {
    font-size: 1.15rem;
    font-weight: 700;
    color: white;
    letter-spacing: -0.01em;
}
.top-bar-left .app-name .accent { color: #f87171; }
.top-bar-left .app-sub {
    font-size: 0.72rem;
    color: #64748b;
    margin-top: 1px;
    font-weight: 400;
}
.status-pill {
    display: flex;
    align-items: center;
    gap: 7px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 100px;
    padding: 6px 16px;
    font-size: 0.75rem;
    font-weight: 500;
    color: #94a3b8;
}
.status-dot {
    width: 7px; height: 7px;
    background: #4ade80;
    border-radius: 50%;
    box-shadow: 0 0 6px #4ade80;
    animation: blink 2s infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.4} }

/* ── CONTENT WRAP ──────────────────────────────────────────────── */
.content-wrap {
    padding: 28px 40px;
}

/* ── CARDS ─────────────────────────────────────────────────────── */
.card {
    background: #ffffff;
    border-radius: 14px;
    border: 1.5px solid #c8d0db;
    box-shadow: 0 2px 8px rgba(0,0,0,0.10), 0 6px 24px rgba(0,0,0,0.08);
    padding: 26px 28px;
    margin-bottom: 16px;
}
.card-header {
    font-size: 0.8rem;
    font-weight: 800;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: #334155;
    margin-bottom: 22px;
    display: flex;
    align-items: center;
    gap: 9px;
}
.card-header-bar {
    display: inline-block;
    width: 4px; height: 15px;
    background: #dc2626;
    border-radius: 2px;
}

/* ── SECTION LABEL ─────────────────────────────────────────────── */
.section-lbl {
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #475569;
    margin: 26px 0 14px 0;
    padding: 9px 12px;
    background: #f8fafc;
    border-left: 3px solid #dc2626;
    border-radius: 0 6px 6px 0;
}

/* ── FIELD LABEL ───────────────────────────────────────────────── */
.field-label {
    font-size: 0.9rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 4px;
    margin-top: 16px;
}
.field-value {
    font-size: 1.15rem;
    font-weight: 800;
    color: #dc2626;
    margin-top: 1px;
    margin-bottom: 4px;
}

/* ── SLIDERS / INPUTS ──────────────────────────────────────────── */
.stSlider > div > div > div > div { background: #dc2626 !important; }
.stSlider [data-baseweb="slider"] > div:first-child {
    background: #e2e8f0 !important;
    height: 6px !important;
    border-radius: 3px !important;
}
[data-testid="stSlider"] span {
    color: #0f172a !important;
    font-weight: 700 !important;
}

/* ── SELECT BOXES ──────────────────────────────────────────────── */
.stSelectbox > div > div {
    background: #f8fafc !important;
    color: #1e293b !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 9px !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    min-height: 44px !important;
}
.stSelectbox > div > div:focus-within {
    border-color: #0d1b2a !important;
    box-shadow: 0 0 0 3px rgba(13,27,42,0.08) !important;
}

/* ── TOGGLE — force dark label text ────────────────────────────── */
[data-testid="stToggle"] label,
[data-testid="stToggle"] label span,
[data-testid="stToggle"] label p,
.stToggle label,
.stToggle label span,
.stToggle p {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: #0f172a !important;
}
/* Toggle ON color */
[data-testid="stToggle"] > label > div[data-checked="true"] { background-color: #dc2626 !important; }

/* ── GLOBAL: ensure no text defaults to white/invisible ─────────── */
.stMarkdown, .stMarkdown p, .stMarkdown span,
label, .stRadio label, .stCheckbox label,
[data-testid="stText"], [data-testid="stMarkdownContainer"] p {
    color: #1e293b !important;
}


/* ── ANALYSE BUTTON ────────────────────────────────────────────── */
.stButton > button {
    background: #0d1b2a !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 0.92rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.09em !important;
    text-transform: uppercase !important;
    padding: 16px 0 !important;
    width: 100% !important;
    transition: all 0.18s ease !important;
    box-shadow: 0 4px 14px rgba(13,27,42,0.30) !important;
    margin-top: 10px !important;
}
/* Force button text white — override broad label rule */
.stButton > button *,
.stButton > button p,
.stButton > button span {
    color: white !important;
}
.stButton > button:hover {
    background: #1a2f48 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(13,27,42,0.25) !important;
}

/* ── TABS ──────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    gap: 6px !important;
    padding: 4px 0 !important;
    border-bottom: 2px solid #e2e8f0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    color: #64748b !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 12px 22px !important;
    border-radius: 0 !important;
    border-bottom: 3px solid transparent !important;
    margin-bottom: -2px !important;
    transition: color 0.15s !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #0d1b2a !important; }
.stTabs [aria-selected="true"] {
    color: #dc2626 !important;
    border-bottom-color: #dc2626 !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ── PLACEHOLDER ───────────────────────────────────────────────── */
.placeholder-card {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 480px;
    text-align: center;
    background: white;
    border-radius: 14px;
    border: 2px dashed #e2e8f0;
}
.placeholder-icon {
    width: 56px; height: 56px;
    background: #f1f5f9;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 16px auto;
    font-size: 1.4rem;
}
.placeholder-title { font-size: 1rem; font-weight: 600; color: #94a3b8; margin: 0 0 6px 0; }
.placeholder-sub   { font-size: 0.82rem; color: #cbd5e1; margin: 0; }

/* ── SNAPSHOT ROW ──────────────────────────────────────────────── */
.snap-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0;
    border-top: 1px solid #f1f5f9;
    margin-top: 12px;
    padding-top: 16px;
}
.snap-item .sv { font-size: 1.5rem; font-weight: 800; color: #0d1b2a; }
.snap-item .sl { font-size: 0.66rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.09em; color: #94a3b8; margin-top: 3px; }

/* ── RISK GRADIENT BAR ─────────────────────────────────────────── */
.risk-bar-wrap {
    margin: 16px 0 10px 0;
    position: relative;
}
.risk-bar-track {
    height: 12px;
    border-radius: 100px;
    background: linear-gradient(to right, #16a34a 0%, #84cc16 25%, #f59e0b 55%, #ef4444 100%);
    position: relative;
    overflow: visible;
}
.risk-bar-marker {
    position: absolute;
    top: -5px;
    width: 22px; height: 22px;
    background: white;
    border: 3px solid #0d1b2a;
    border-radius: 50%;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    transform: translateX(-50%);
    transition: left 0.4s ease;
}
.risk-bar-labels {
    display: flex;
    justify-content: space-between;
    margin-top: 7px;
    font-size: 0.68rem;
    font-weight: 600;
    color: #94a3b8;
    letter-spacing: 0.06em;
}
.risk-pct {
    font-size: 3.2rem;
    font-weight: 900;
    line-height: 1;
    letter-spacing: -0.03em;
}
.risk-pct-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-bottom: 6px;
}
.risk-tier-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    border-radius: 100px;
    padding: 6px 16px;
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 10px;
}
.tier-low    { background: #dcfce7; color: #15803d; }
.tier-medium { background: #fef3c7; color: #b45309; }
.tier-high   { background: #fee2e2; color: #b91c1c; }
.tier-dot    { width: 8px; height: 8px; border-radius: 50%; }

/* ── AI RECOMMENDATION CARD ────────────────────────────────────── */
.rec-card {
    border-radius: 12px;
    padding: 22px 24px;
    margin-top: 4px;
}
.rec-low    { background: #f0fdf4; border: 1.5px solid #86efac; }
.rec-medium { background: #fffbeb; border: 1.5px solid #fcd34d; }
.rec-high   { background: #fff1f2; border: 1.5px solid #fca5a5; }
.rec-tag {
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    opacity: 0.6;
    margin-bottom: 4px;
}
.rec-action {
    font-size: 1.55rem;
    font-weight: 800;
    margin-bottom: 12px;
    letter-spacing: -0.01em;
}
.rec-divider {
    height: 1px;
    background: rgba(0,0,0,0.06);
    margin: 12px 0;
}
.rec-reason-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    opacity: 0.5;
    margin-bottom: 5px;
}
.rec-reason-text {
    font-size: 0.88rem;
    line-height: 1.65;
    opacity: 0.85;
}
.rec-next-step {
    margin-top: 14px;
    padding: 11px 14px;
    background: rgba(0,0,0,0.04);
    border-radius: 8px;
    font-size: 0.82rem;
    font-weight: 600;
    opacity: 0.8;
}

/* ── WHAT-IF ───────────────────────────────────────────────────── */
.whatif-result-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 18px;
    margin-top: 14px;
}
.whatif-pct { font-size: 1.6rem; font-weight: 800; color: #0d1b2a; }
.whatif-delta { font-size: 0.82rem; font-weight: 700; }

/* ── POLICY ────────────────────────────────────────────────────── */
.policy-row {
    display: flex;
    gap: 10px;
    padding: 12px 0;
    border-bottom: 1px solid #f1f5f9;
    font-size: 0.86rem;
    color: #374151;
    line-height: 1.55;
    align-items: flex-start;
}
.policy-row:last-child { border-bottom: none; }
.policy-arrow { color: #dc2626; font-size: 0.75rem; margin-top: 2px; flex-shrink: 0; }
.policy-row b { color: #0d1b2a; }

/* ── DISCLAIMER ─────────────────────────────────────────────────── */
.disclaimer {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 0.8rem;
    color: #64748b;
    line-height: 1.6;
    margin-top: 14px;
}
.disclaimer b { color: #374151; }

/* ── BATCH KPIs ─────────────────────────────────────────────────── */
.bkpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 20px;
}
.bkpi {
    background: white;
    border-radius: 12px;
    border: 1px solid #e8edf2;
    padding: 20px 22px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.bkpi .bv { font-size: 2rem; font-weight: 800; color: #0d1b2a; }
.bkpi .bl { font-size: 0.68rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.09em; color: #94a3b8; margin-top: 4px; }
.bkpi.high   { border-left: 4px solid #dc2626; }
.bkpi.medium { border-left: 4px solid #f59e0b; }
.bkpi.low    { border-left: 4px solid #16a34a; }
.bkpi.info   { border-left: 4px solid #3b82f6; }

/* ── BATCH UPLOAD AREA ──────────────────────────────────────────── */
.upload-hint {
    font-size: 0.82rem;
    color: #64748b;
    margin-bottom: 10px;
}
.required-cols {
    font-size: 0.78rem;
    color: #94a3b8;
    background: #f8fafc;
    border: 1px dashed #e2e8f0;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 14px;
    font-family: 'Inter', monospace;
}

/* ── DATAFRAME TWEAKS ───────────────────────────────────────────── */
[data-testid="stDataFrame"] { border-radius: 10px !important; overflow: hidden; }

/* ── GENERAL TYPOGRAPHY ─────────────────────────────────────────── */
p, li, span { font-size: 0.9rem; }
h3 { font-size: 1.05rem; font-weight: 700; color: #0d1b2a; }
</style>
""", unsafe_allow_html=True)

# ── Artifact paths ────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH   = os.path.join(BASE_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "feature_cols.pkl")

@st.cache_resource
def load_artifacts():
    m  = joblib.load(MODEL_PATH)
    sc = joblib.load(SCALER_PATH)
    fc = joblib.load(FEATURES_PATH)
    return m, sc, fc

if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, FEATURES_PATH]):
    st.error("Model artifacts not found. Run: python model_brain.py")
    st.stop()

model, scaler, feature_cols = load_artifacts()


# ── Helper functions ──────────────────────────────────────────────
def predict_prob(age, gender, awaiting, sms, hipert, diab, schol, hcap, prev_miss):
    h = {f"Handicap_{i}": 0 for i in range(5)}
    h[f"Handicap_{hcap}"] = 1
    row = {
        "Age": age, "AwaitingTime": awaiting,
        "SMS_received": 1 if sms else 0,
        "Hipertension": 1 if hipert else 0,
        "Diabetes": 1 if diab else 0,
        "Scholarship": 1 if schol else 0,
        "Num_App_Missed": prev_miss,
        "Gender_F": 1 if gender == "Female" else 0,
        "Gender_M": 1 if gender == "Male" else 0,
        **h
    }
    df_row = pd.DataFrame([row])
    for col in feature_cols:
        if col not in df_row.columns:
            df_row[col] = 0
    df_row  = df_row[feature_cols]
    scaled  = scaler.transform(df_row)
    return float(model.predict_proba(scaled)[0][1])

def risk_tier(prob):
    if prob < 0.30:   return "LOW",    "#16a34a", "tier-low"
    elif prob < 0.55: return "MEDIUM", "#f59e0b", "tier-medium"
    else:             return "HIGH",   "#dc2626", "tier-high"

def rec_action(prob):
    if prob < 0.30:
        return (
            "STANDARD CONFIRM",
            "rec-low",
            "This patient shows a low probability of missing their appointment. Standard automated confirmation is sufficient.",
            "Ensure the appointment reminder is dispatched via the automated system at least 48 hours prior."
        )
    elif prob < 0.55:
        return (
            "SEND SMS REMINDER",
            "rec-medium",
            "Moderate risk of non-attendance detected. An additional personalised SMS reminder should be sent 24 hours before the appointment.",
            "If the slot is high-value or critical, follow up with a brief courtesy call to confirm attendance."
        )
    else:
        return (
            "CALL PATIENT NOW",
            "rec-high",
            "High probability of no-show detected. A direct call to the patient is strongly recommended to confirm attendance intent.",
            "If no confirmation is received within 4 hours of the appointment, activate the standby overbooking protocol."
        )

def preprocess_batch(df_raw):
    required = {"Age","Gender","Hipertension","Diabetes","Scholarship",
                 "SMS_received","Handcap","ScheduledDay","AppointmentDay","PatientId","No-show"}
    missing = required - set(df_raw.columns)
    if missing:
        st.error(f"CSV missing columns: {missing}")
        return None, None
    df = df_raw.copy()
    df["No-show"] = df["No-show"].replace({"No":0,"Yes":1})
    df["Handcap"] = pd.Categorical(df["Handcap"])
    hd = pd.get_dummies(df["Handcap"], prefix="Handicap")
    df = pd.concat([df, hd], axis=1)
    for i in range(5):
        if f"Handicap_{i}" not in df.columns:
            df[f"Handicap_{i}"] = 0
    df = df[(df.Age >= 0) & (df.Age <= 100)].copy()
    df["ScheduledDay"]   = pd.to_datetime(df["ScheduledDay"])
    df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"])
    df["AwaitingTime"]   = (df["AppointmentDay"].sub(df["ScheduledDay"]) / np.timedelta64(1,"D")).abs()
    ds = df.sort_values(["PatientId","AppointmentDay"]).copy()
    ds["Num_App_Missed"] = ds.groupby("PatientId")["No-show"].transform(
        lambda x: x.shift().fillna(0).cumsum())
    df["Num_App_Missed"] = ds["Num_App_Missed"]
    X = df[["Gender","Diabetes","Hipertension","Scholarship","SMS_received",
             "Handicap_0","Handicap_1","Handicap_2","Handicap_3","Handicap_4",
             "Num_App_Missed","Age","AwaitingTime"]]
    X = pd.get_dummies(X)
    for col in feature_cols:
        if col not in X.columns: X[col] = 0
    X = X[feature_cols]
    return scaler.transform(X), df.index.tolist()


# ═══════════════════════════════════════════════════════════════════
# TOP BAR
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="top-bar">
  <div class="top-bar-left">
    <div class="app-name"><span class="accent">&#9670;</span> Patient Attendance Intelligence Console</div>
    <div class="app-sub">AI-Powered Appointment Risk Engine</div>
  </div>
  <div class="status-pill">
    <span class="status-dot"></span>
    Model Status: Active
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════
st.markdown('<div style="margin-left: -3rem; margin-right: -3rem; padding: 0 40px; background: white; border-bottom: 1px solid #e2e8f0;">', unsafe_allow_html=True)
tab_single, tab_batch = st.tabs(["  Single Patient Analysis  ", "  Batch Analysis  "])
st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# SINGLE PATIENT TAB
# ═══════════════════════════════════════════════════════════════════
with tab_single:
    st.markdown('<div class="content-wrap">', unsafe_allow_html=True)
    col_left, col_right = st.columns([4, 6], gap="large")

    # ── LEFT: PATIENT PROFILE FORM ────────────────────────────────
    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header"><span class="card-header-bar"></span>PATIENT PROFILE</div>', unsafe_allow_html=True)

        # PERSONAL DETAILS
        st.markdown('<div class="section-lbl">Personal Details</div>', unsafe_allow_html=True)

        st.markdown('<div class="field-label">Age</div>', unsafe_allow_html=True)
        age = st.slider("Age", 0, 100, 40, key="age", label_visibility="collapsed")
        st.markdown(f'<div class="field-value">{age} years</div>', unsafe_allow_html=True)

        st.markdown('<div class="field-label">Gender</div>', unsafe_allow_html=True)
        gender = st.selectbox("Gender", ["Female", "Male"], key="gender", label_visibility="collapsed")

        # APPOINTMENT DETAILS
        st.markdown('<div class="section-lbl">Appointment Details</div>', unsafe_allow_html=True)

        st.markdown('<div class="field-label">Lead Time — Days between booking and appointment</div>', unsafe_allow_html=True)
        awaiting = st.slider("Lead Time", 0, 180, 7, key="awaiting", label_visibility="collapsed")
        st.markdown(f'<div class="field-value">{awaiting} days</div>', unsafe_allow_html=True)

        st.markdown('<div class="field-label">SMS Reminder</div>', unsafe_allow_html=True)
        sms = st.toggle("SMS reminder was sent to patient", key="sms", value=False)

        # MEDICAL HISTORY
        st.markdown('<div class="section-lbl">Medical History</div>', unsafe_allow_html=True)

        hipert = st.toggle("Hypertension diagnosed", key="hipert", value=False)
        diab   = st.toggle("Diabetes diagnosed", key="diab", value=False)
        schol  = st.toggle("Scholarship / Social Insurance", key="schol", value=False)

        st.markdown('<div class="field-label">Disability Level</div>', unsafe_allow_html=True)
        hcap = st.selectbox("Disability Level", [0,1,2,3,4], key="hcap",
                             label_visibility="collapsed",
                             format_func=lambda x: f"Level {x} — No disability" if x == 0 else f"Level {x}")

        st.markdown('<div class="field-label">Previous No-Shows by this patient</div>', unsafe_allow_html=True)
        prev_miss = st.slider("Previous No-Shows", 0, 10, 0, key="prev_miss", label_visibility="collapsed")
        st.markdown(f'<div class="field-value">{prev_miss} missed appointments</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)  # close card

        analyse = st.button("Analyse Appointment Risk", key="analyse_btn")

    # Store result
    if analyse:
        prob = predict_prob(age, gender, awaiting, sms, hipert, diab, schol, hcap, prev_miss)
        st.session_state.result = {
            "prob": prob, "age": age, "gender": gender, "awaiting": awaiting, "sms": sms,
            "hipert": hipert, "diab": diab, "schol": schol, "hcap": hcap, "prev_miss": prev_miss
        }

    # ── RIGHT: RESULTS PANEL ──────────────────────────────────────
    with col_right:

        if "result" not in st.session_state:
            st.markdown("""
            <div class="placeholder-card">
              <div class="placeholder-icon">&#9670;</div>
              <div class="placeholder-title">No Analysis Run Yet</div>
              <div class="placeholder-sub">Complete the patient profile on the left and click Analyse</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            r    = st.session_state.result
            prob = r["prob"]
            pct  = prob * 100
            label, color, tier_cls = risk_tier(prob)
            action, rec_cls, reason, next_step = rec_action(prob)

            # ── PATIENT SNAPSHOT ──────────────────────────────────
            st.markdown('<div class="card" style="padding-bottom: 20px;">', unsafe_allow_html=True)
            st.markdown('<div class="card-header"><span class="card-header-bar"></span>PATIENT SNAPSHOT</div>', unsafe_allow_html=True)

            cond_parts = []
            if r["hipert"]: cond_parts.append("Hypertension")
            if r["diab"]:   cond_parts.append("Diabetes")
            if r["schol"]:  cond_parts.append("Scholarship")
            cond_str = " / ".join(cond_parts) if cond_parts else "None"

            st.markdown(f"""
            <div class="snap-grid">
              <div class="snap-item">
                <div class="sv">{r['age']}y</div>
                <div class="sl">Age</div>
              </div>
              <div class="snap-item">
                <div class="sv">{r['awaiting']}d</div>
                <div class="sl">Lead Time</div>
              </div>
              <div class="snap-item">
                <div class="sv">{"Sent" if r['sms'] else "None"}</div>
                <div class="sl">SMS Status</div>
              </div>
              <div class="snap-item">
                <div class="sv">{r['prev_miss']}</div>
                <div class="sl">Prior No-Shows</div>
              </div>
            </div>
            <div style="margin-top: 12px; font-size: 0.78rem; color: #94a3b8; padding-top: 10px; border-top: 1px solid #f1f5f9;">
              Conditions: {cond_str} &nbsp;|&nbsp; Disability: Level {r['hcap']} &nbsp;|&nbsp; Gender: {r['gender']}
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # ── RISK ANALYSIS ─────────────────────────────────────
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header"><span class="card-header-bar"></span>RISK ANALYSIS</div>', unsafe_allow_html=True)

            marker_pos = min(max(pct, 2), 98)
            st.markdown(f"""
            <div class="risk-pct-label">No-Show Probability</div>
            <div class="risk-pct" style="color: {color};">{pct:.1f}%</div>

            <div class="risk-bar-wrap">
              <div class="risk-bar-track">
                <div class="risk-bar-marker" style="left: {marker_pos}%;"></div>
              </div>
              <div class="risk-bar-labels">
                <span>LOW RISK</span>
                <span>MODERATE</span>
                <span>HIGH RISK</span>
              </div>
            </div>

            <div class="risk-tier-badge {tier_cls}">
              <div class="tier-dot" style="background:{color};"></div>
              {label} RISK — Patient likely {"will" if prob >= 0.55 else ("may" if prob >= 0.30 else "will not")} miss this appointment
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # ── AI RECOMMENDATION ─────────────────────────────────
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header"><span class="card-header-bar"></span>AI RECOMMENDATION</div>', unsafe_allow_html=True)

            st.markdown(f"""
            <div class="rec-card {rec_cls}">
              <div class="rec-tag">Primary Action</div>
              <div class="rec-action">{action}</div>
              <div class="rec-divider"></div>
              <div class="rec-reason-label">Reason</div>
              <div class="rec-reason-text">{reason}</div>
              <div class="rec-next-step">
                Suggested Next Step: {next_step}
              </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # ── SCENARIO SIMULATION ───────────────────────────────
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header"><span class="card-header-bar"></span>SCENARIO SIMULATION</div>', unsafe_allow_html=True)
            st.markdown('<div style="font-size:0.85rem; color:#64748b; margin-bottom:14px;">Adjust variables to see how the risk score changes.</div>', unsafe_allow_html=True)

            wi1, wi2, wi3 = st.columns(3)
            with wi1:
                wi_await = st.slider("Lead Time (days)", 0, 180, r["awaiting"], key="wi_await")
            with wi2:
                wi_sms = st.toggle("SMS Sent", key="wi_sms", value=r["sms"])
            with wi3:
                wi_prev = st.slider("Prior No-Shows", 0, 10, r["prev_miss"], key="wi_prev")

            wi_prob   = predict_prob(r["age"], r["gender"], wi_await, wi_sms,
                                     r["hipert"], r["diab"], r["schol"], r["hcap"], wi_prev)
            wi_pct    = wi_prob * 100
            wi_lbl, wi_col, _ = risk_tier(wi_prob)
            wi_action, _, _, _ = rec_action(wi_prob)
            delta     = pct - wi_pct
            delta_str = f"-{abs(delta):.1f}pp" if delta > 0 else (f"+{abs(delta):.1f}pp" if delta < 0 else "No change")
            d_color   = "#16a34a" if delta > 0 else ("#dc2626" if delta < 0 else "#64748b")

            st.markdown(f"""
            <div class="whatif-result-bar">
              <div>
                <div style="font-size:0.68rem;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:0.09em;margin-bottom:3px;">Simulated Risk</div>
                <div class="whatif-pct" style="color:{wi_col};">{wi_pct:.1f}%</div>
              </div>
              <div style="text-align:right;">
                <div class="whatif-delta" style="color:{d_color};">{delta_str} vs. original</div>
                <div style="margin-top:5px; font-size:0.78rem; font-weight:700; background:{wi_col}; color:white;
                     border-radius:100px; padding:4px 14px; display:inline-block;">{wi_action}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # ── HOSPITAL POLICY ───────────────────────────────────
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header"><span class="card-header-bar"></span>HOSPITAL POLICY REFERENCES</div>', unsafe_allow_html=True)
            policies = [
                ("<b>Attendance Management Protocol v3.2</b>", "Outbound call required for all patients with no-show probability above 55%."),
                ("<b>SMS Reminder Policy</b>", "Automated SMS must be sent 48 hours before appointment. Secondary SMS at 24 hours for medium-risk."),
                ("<b>Slot Overbooking Guideline</b>", "Standby patients may be activated up to 4 hours before appointment for confirmed high-risk slots."),
                ("<b>Data Governance — Patient Contact</b>", "All outreach attempts must be logged in the Patient Contact Register."),
            ]
            for title, body in policies:
                st.markdown(f"""
                <div class="policy-row">
                  <span class="policy-arrow">&#9658;</span>
                  <div>{title}<br><span style="font-weight:400;color:#6b7280;">{body}</span></div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # ── EXPORT ────────────────────────────────────────────
            today_str = datetime.now().strftime("%Y-%m-%d %H:%M")
            cond_str2 = ", ".join(cond_parts) if cond_parts else "None"
            export_txt = "\n".join([
                "=" * 60,
                "  PATIENT ATTENDANCE INTELLIGENCE CONSOLE",
                f"  Generated: {today_str}",
                "=" * 60, "",
                "PATIENT DETAILS",
                f"  Age: {r['age']}  |  Gender: {r['gender']}  |  Lead Time: {r['awaiting']} days",
                f"  SMS Reminder: {'Sent' if r['sms'] else 'Not Sent'}  |  Prior No-Shows: {r['prev_miss']}",
                f"  Conditions: {cond_str2}  |  Disability: Level {r['hcap']}",
                "",
                "RISK ASSESSMENT",
                f"  No-Show Probability: {pct:.1f}%",
                f"  Risk Tier: {label}",
                "",
                "AI RECOMMENDATION",
                f"  Action: {action}",
                f"  Reason: {reason}",
                f"  Next Step: {next_step}",
                "",
                "=" * 60,
                "DISCLAIMER: For operational use by authorised clinical staff only.",
                "=" * 60,
            ])
            st.download_button(
                label="Export Patient Action Plan (.txt)",
                data=export_txt,
                file_name=f"patient_action_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                key="export_single",
            )

            st.markdown("""
            <div class="disclaimer">
              <b>Notice:</b> This decision support output is generated by a machine learning model trained on historical appointment data.
              Predictions are probabilistic and should be reviewed by qualified administrative staff before operational action is taken.
              The model does not account for real-time patient communication or clinical urgency.
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # close content-wrap


# ═══════════════════════════════════════════════════════════════════
# BATCH ANALYSIS TAB
# ═══════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown('<div class="content-wrap">', unsafe_allow_html=True)

    # UPLOAD CARD
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header"><span class="card-header-bar"></span>UPLOAD APPOINTMENT SCHEDULE</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="upload-hint">Upload a CSV file exported from your Hospital Management System in KaggleV2 format.</div>
    <div class="required-cols">Required columns: PatientId, Gender, Age, Hipertension, Diabetes, Scholarship, Handcap, SMS_received, ScheduledDay, AppointmentDay, No-show</div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed", key="batch_csv")
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded:
        df_raw = pd.read_csv(uploaded)
        X_scaled, idx = preprocess_batch(df_raw)

        if X_scaled is not None:
            probs = model.predict_proba(X_scaled)[:,1]
            df_out = df_raw.loc[idx].copy()
            df_out["No-Show Probability (%)"] = (probs * 100).round(1)
            df_out["Risk Tier"] = ["HIGH" if p >= 0.55 else ("MEDIUM" if p >= 0.30 else "LOW") for p in probs]

            n_total  = len(df_out)
            n_high   = (df_out["Risk Tier"] == "HIGH").sum()
            n_medium = (df_out["Risk Tier"] == "MEDIUM").sum()
            est_miss = int(round(probs.mean() * n_total))
            standby  = max(0, int(round(n_high * 0.5)))

            # KPI Row
            st.markdown(f"""
            <div class="bkpi-grid">
              <div class="bkpi info">
                <div class="bv">{n_total:,}</div>
                <div class="bl">Total Appointments</div>
              </div>
              <div class="bkpi high">
                <div class="bv">{n_high:,}</div>
                <div class="bl">High-Risk Patients</div>
              </div>
              <div class="bkpi medium">
                <div class="bv">{est_miss:,}</div>
                <div class="bl">Estimated No-Shows</div>
              </div>
              <div class="bkpi low">
                <div class="bv">{standby:,}</div>
                <div class="bl">Suggested Standby Slots</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Risk Distribution Chart + Action Summary
            col_ch, col_act = st.columns([6, 4], gap="large")

            with col_ch:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-header"><span class="card-header-bar"></span>RISK DISTRIBUTION</div>', unsafe_allow_html=True)
                tier_counts = df_out["Risk Tier"].value_counts().reindex(["HIGH","MEDIUM","LOW"], fill_value=0)

                fig = go.Figure(go.Bar(
                    x=["High Risk", "Medium Risk", "Low Risk"],
                    y=[tier_counts.get("HIGH",0), tier_counts.get("MEDIUM",0), tier_counts.get("LOW",0)],
                    marker_color=["#dc2626", "#f59e0b", "#16a34a"],
                    marker_line_width=0,
                ))
                fig.update_layout(
                    height=240,
                    margin=dict(t=10, b=10, l=0, r=0),
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    xaxis=dict(tickfont=dict(size=12, family="Inter"), gridcolor="#f1f5f9"),
                    yaxis=dict(tickfont=dict(size=11, family="Inter"), gridcolor="#f1f5f9"),
                    font=dict(family="Inter"),
                    bargap=0.35,
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col_act:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-header"><span class="card-header-bar"></span>OPERATIONAL ACTIONS</div>', unsafe_allow_html=True)
                actions = [
                    (f"{n_high} patients", "HIGH RISK", "#dc2626", "Call immediately to confirm attendance"),
                    (f"{n_medium} patients", "MEDIUM RISK", "#f59e0b", "Send personalised SMS reminder"),
                    (f"{(df_out['Risk Tier']=='LOW').sum()} patients", "LOW RISK", "#16a34a", "Standard confirmation — no action needed"),
                    (f"{standby} slots", "STANDBY NEEDED", "#3b82f6", "Activate standby overbooking list"),
                ]
                for stat, tier, col, desc in actions:
                    st.markdown(f"""
                    <div style="display:flex; justify-content:space-between; align-items:flex-start;
                         padding: 12px 0; border-bottom: 1px solid #f1f5f9;">
                      <div>
                        <div style="font-size:0.72rem; font-weight:700; text-transform:uppercase;
                             letter-spacing:0.09em; color:{col}; margin-bottom:2px;">{tier}</div>
                        <div style="font-size:0.84rem; color:#374151;">{desc}</div>
                      </div>
                      <div style="font-size:1.1rem; font-weight:800; color:#0d1b2a; white-space:nowrap; margin-left:12px;">{stat}</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # High Risk Patient List
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header"><span class="card-header-bar"></span>HIGH-RISK PATIENT LIST</div>', unsafe_allow_html=True)
            display_cols = ["PatientId","Age","Gender","No-Show Probability (%)","Risk Tier"]
            avail = [c for c in display_cols if c in df_out.columns]
            df_high = (df_out[df_out["Risk Tier"] == "HIGH"][avail]
                       .sort_values("No-Show Probability (%)", ascending=False)
                       .reset_index(drop=True))
            if len(df_high) > 0:
                st.dataframe(df_high, use_container_width=True, height=280)
            else:
                st.success("No high-risk patients found in this upload.")
            st.markdown('</div>', unsafe_allow_html=True)

            # Export
            csv_export = df_out.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Export Full Risk Report (.csv)",
                data=csv_export,
                file_name=f"batch_risk_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                key="export_batch",
            )

            st.markdown("""
            <div class="disclaimer">
              <b>Notice:</b> Predictions are based on historical attendance pattern modelling.
              Revenue estimates assume a fixed per-appointment value and are indicative only.
              Operational decisions should be reviewed by qualified administrative staff.
            </div>
            """, unsafe_allow_html=True)

    else:
        # No file uploaded — show empty state
        st.markdown("""
        <div class="placeholder-card" style="min-height: 380px;">
          <div class="placeholder-icon">&#8659;</div>
          <div class="placeholder-title">Upload Your Appointment Schedule</div>
          <div class="placeholder-sub">Drag & drop or click the upload area above to analyse your batch data</div>
          <div style="margin-top: 22px; padding: 14px 24px; background: #f8fafc; border-radius: 10px;
               border: 1px solid #e2e8f0; font-size: 0.82rem; color: #64748b; max-width: 400px; text-align:left;">
            After upload, you will see:<br>
            &nbsp;&#8250; &nbsp;Total appointments and risk counts<br>
            &nbsp;&#8250; &nbsp;Risk distribution chart<br>
            &nbsp;&#8250; &nbsp;Recommended operational actions<br>
            &nbsp;&#8250; &nbsp;Sortable high-risk patient list<br>
            &nbsp;&#8250; &nbsp;Exportable risk report
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # close content-wrap
