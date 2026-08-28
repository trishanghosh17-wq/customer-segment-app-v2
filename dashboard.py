"""
Customer Segment Predictor — Enhanced Dashboard
================================================
A richer, more engaging Streamlit app for RFM-based customer segmentation.

Drop your real model in models/customer_segment_classifier.joblib.
If it's not found, a synthetic demo model is trained on the fly so the
app is fully explorable out of the box.
"""

from pathlib import Path
from datetime import datetime
import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Customer Segment Predictor",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODELS = Path(__file__).parent / "models"
FEATURES = ["Recency", "Frequency", "Monetary"]

SEGMENT_COLORS = {
    "Champions": "#00CEC9",
    "Loyal Customers": "#6C5CE7",
    "New / Occasional": "#FDCB6E",
    "Hibernating": "#8B93A7",
}
SEGMENT_INFO = {
    "Champions": {
        "emoji": "🏆",
        "desc": "Recent, frequent, high spend — your most valuable customers.",
        "action": "Reward with loyalty perks and early access to new products.",
        "priority": "High",
    },
    "Loyal Customers": {
        "emoji": "💙",
        "desc": "Solid frequency and spend, moderately recent.",
        "action": "Good candidates for upsell/cross-sell and bundle offers.",
        "priority": "Medium-High",
    },
    "New / Occasional": {
        "emoji": "🌱",
        "desc": "Bought recently but with few orders so far.",
        "action": "Target with second-purchase incentives to build the habit.",
        "priority": "Medium",
    },
    "Hibernating": {
        "emoji": "😴",
        "desc": "Long time since last purchase, low historical spend.",
        "action": "Low-cost reactivation campaigns only.",
        "priority": "Low",
    },
}
SEGMENT_ORDER = list(SEGMENT_INFO.keys())

# ============================================================
# STYLE — animated design system
# ============================================================
st.markdown("""
<style>
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes floatShape {
    0%, 100% { transform: translateY(0) rotate(0deg); }
    50% { transform: translateY(-16px) rotate(8deg); }
}
@keyframes popIn {
    0% { opacity: 0; transform: scale(.85); }
    70% { transform: scale(1.03); }
    100% { opacity: 1; transform: scale(1); }
}
@keyframes shimmer {
    0% { background-position: -400px 0; }
    100% { background-position: 400px 0; }
}
@keyframes pulseRing {
    0% { box-shadow: 0 0 0 0 rgba(108,92,231,.45); }
    70% { box-shadow: 0 0 0 12px rgba(108,92,231,0); }
    100% { box-shadow: 0 0 0 0 rgba(108,92,231,0); }
}
@keyframes auroraShift {
    0%   { background-position: 0% 0%, 100% 0%, 50% 100%, 0% 0%; }
    50%  { background-position: 100% 100%, 0% 100%, 50% 0%, 100% 100%; }
    100% { background-position: 0% 0%, 100% 0%, 50% 100%, 0% 0%; }
}
@keyframes twinkle {
    0%, 100% { opacity: .15; transform: scale(.7); }
    50% { opacity: 1; transform: scale(1.3); }
}
@keyframes idleFloat {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-7px); }
}
@keyframes wordIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes ringDraw {
    from { stroke-dashoffset: var(--ring-full); }
    to   { stroke-dashoffset: var(--ring-offset); }
}
@keyframes glowPulse {
    0%, 100% { filter: drop-shadow(0 0 2px currentColor); }
    50% { filter: drop-shadow(0 0 9px currentColor); }
}

/* ---------- Whole-app animated aurora background ---------- */
[data-testid="stAppViewContainer"] {
    background-image:
        radial-gradient(circle at 15% 20%, rgba(108,92,231,.16), transparent 42%),
        radial-gradient(circle at 85% 10%, rgba(0,206,201,.14), transparent 42%),
        radial-gradient(circle at 50% 100%, rgba(253,203,110,.13), transparent 45%),
        radial-gradient(circle at 0% 100%, rgba(108,92,231,.10), transparent 40%);
    background-size: 180% 180%, 180% 180%, 200% 200%, 160% 160%;
    animation: auroraShift 22s ease-in-out infinite;
}

/* ---------- Hero header ---------- */
.hero {
    position: relative; overflow: hidden;
    padding: 2.1rem 2.3rem; border-radius: 20px; margin-bottom: 1.3rem;
    background: linear-gradient(120deg, #6C5CE7, #00CEC9, #FDCB6E, #6C5CE7);
    background-size: 300% 300%;
    animation: gradientShift 10s ease infinite;
    color: white; box-shadow: 0 12px 30px rgba(108,92,231,.25);
}
.hero::before, .hero::after {
    content: ""; position: absolute; border-radius: 50%;
    background: rgba(255,255,255,.16); animation: floatShape 7s ease-in-out infinite;
}
.hero::before { width: 130px; height: 130px; top: -50px; right: 70px; }
.hero::after  { width: 80px; height: 80px; bottom: -25px; right: 220px; animation-delay: 1.8s; }
.hero h1 { margin: 0; font-size: 2.1rem; animation: fadeInUp .6s ease both; position: relative; z-index: 1; }
.hero .subtitle { margin: .45rem 0 0 0; opacity: .95; position: relative; z-index: 1; }
.hero .word { display: inline-block; opacity: 0; animation: wordIn .5s ease forwards; }
.spark {
    position: absolute; width: 5px; height: 5px; border-radius: 50%;
    background: white; animation: twinkle 2.6s ease-in-out infinite;
}

/* ---------- Reusable entrance animation ---------- */
.animate-in { animation: fadeInUp .55s cubic-bezier(.2,.8,.2,1) both; }

/* ---------- KPI / metric cards ---------- */
.kpi-card {
    background: linear-gradient(145deg, #ffffff12, #ffffff05);
    border: 1px solid #ffffff22; border-radius: 14px; padding: 1rem 1rem;
    text-align: center; transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
    animation: fadeInUp .5s ease both;
    transform-style: preserve-3d; perspective: 600px;
}
.kpi-card:hover { transform: perspective(600px) rotateX(4deg) rotateY(-4deg) translateY(-6px); box-shadow: 0 14px 28px rgba(0,0,0,.2); border-color: #ffffff44; }
.kpi-card .kpi-label { font-size: .78rem; opacity: .7; letter-spacing: .02em; text-transform: uppercase; }
.kpi-card .kpi-value { font-size: 1.7rem; font-weight: 700; margin-top: .15rem; }

/* ---------- Prediction result card ---------- */
.result-card {
    border-radius: 18px; padding: 1.5rem 1.7rem; color: white; position: relative;
    overflow: hidden; box-shadow: 0 14px 34px rgba(0,0,0,.25);
    animation: popIn .55s cubic-bezier(.2,.85,.3,1.2) both, idleFloat 4.5s ease-in-out infinite .6s;
    display: flex; justify-content: space-between; align-items: center; gap: 1.2rem; flex-wrap: wrap;
}
.result-card::after {
    content: ""; position: absolute; inset: 0; pointer-events: none;
    background: radial-gradient(circle at 90% 10%, rgba(255,255,255,.25), transparent 55%);
}
.result-badge {
    display: inline-block; padding: .35rem .9rem; border-radius: 999px;
    background: rgba(255,255,255,.22); font-weight: 700; font-size: .85rem;
    letter-spacing: .02em; animation: pulseRing 2.4s ease infinite;
}
.result-title { font-size: 1.5rem; font-weight: 800; margin: .55rem 0 .2rem 0; }
.result-desc { opacity: .95; margin-bottom: .1rem; }
.confidence-track {
    background: rgba(255,255,255,.28); border-radius: 999px; height: 16px;
    overflow: hidden; margin-top: .8rem; position: relative;
}
.confidence-fill {
    height: 100%; border-radius: 999px;
    background: linear-gradient(90deg, rgba(255,255,255,.55) 25%, rgba(255,255,255,.95) 50%, rgba(255,255,255,.55) 75%);
    background-size: 400px 100%;
    animation: shimmer 2.2s linear infinite;
    transition: width 1.1s cubic-bezier(.2,.85,.2,1);
}
.confidence-label { font-size: .78rem; opacity: .9; margin-top: .3rem; text-align: right; }
.confidence-ring circle.ring-bg { fill: none; stroke: rgba(255,255,255,.25); }
.confidence-ring circle.ring-fg {
    fill: none; stroke: white; stroke-linecap: round;
    animation: ringDraw 1.3s cubic-bezier(.2,.85,.2,1) forwards, glowPulse 2.4s ease-in-out infinite 1.3s;
    color: white;
}
.confidence-ring text { fill: white; font-weight: 800; }

/* ---------- Segment legend chips ---------- */
.segment-chip {
    border-radius: 14px; padding: .9rem 1rem; text-align: left;
    transition: transform .25s ease, box-shadow .25s ease; animation: fadeInUp .5s ease both;
    border: 1px solid #ffffff1f; perspective: 600px;
}
.segment-chip:hover { transform: perspective(600px) rotateX(3deg) rotateY(-3deg) translateY(-4px) scale(1.015); box-shadow: 0 12px 24px rgba(0,0,0,.18); }
.segment-chip h4 { margin: 0 0 .25rem 0; }
.segment-chip p { margin: 0; font-size: .84rem; opacity: .92; }

/* ---------- Interactive widgets ---------- */
.stButton > button {
    transition: transform .18s ease, box-shadow .18s ease !important;
    border-radius: 10px !important;
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 18px rgba(108,92,231,.35); }
.stButton > button:active { transform: translateY(0) scale(.96); }

button[kind="primary"] {
    background: linear-gradient(120deg, #6C5CE7, #00CEC9, #6C5CE7) !important;
    background-size: 220% 220% !important;
    animation: gradientShift 4s ease infinite !important;
    border: none !important;
}
button[kind="primary"]:hover { box-shadow: 0 0 0 4px rgba(108,92,231,.25), 0 10px 22px rgba(108,92,231,.4) !important; }

[data-baseweb="tab-list"] button { transition: transform .18s ease, color .18s ease; }
[data-baseweb="tab-list"] button:hover { transform: translateY(-2px); }
[data-baseweb="tab-highlight"] { transition: all .3s cubic-bezier(.2,.85,.2,1) !important; }

div[data-testid="stMetricValue"] { animation: fadeInUp .5s ease both; }
div[data-testid="stExpander"] { transition: box-shadow .2s ease; }
img, .stPlotlyChart { animation: fadeInUp .6s ease both; }
</style>
""", unsafe_allow_html=True)


def confetti_burst(pieces=140, colors=None):
    """Fires a lightweight client-side confetti animation (canvas-confetti via CDN)."""
    color_js = ""
    if colors:
        js_list = ", ".join(f"'{c}'" for c in colors)
        color_js = f", colors: [{js_list}]"
    components.html(f"""
        <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.9.2/dist/confetti.browser.min.js"></script>
        <script>
        try {{
          confetti({{ particleCount: {pieces}, spread: 90, origin: {{ y: 0.4 }}{color_js} }});
        }} catch (e) {{}}
        </script>
    """, height=0, width=0)


def kpi_card(label, value, delay=0):
    st.markdown(f"""
    <div class="kpi-card" style="animation-delay:{delay}s">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def confidence_ring(pct, color, size=92, stroke=9):
    """Animated SVG ring that draws itself in to show a confidence percentage."""
    radius = (size - stroke) / 2
    circumference = 2 * 3.14159265 * radius
    offset = circumference * (1 - pct)
    st.markdown(f"""
    <div style="display:flex; justify-content:center;">
    <svg class="confidence-ring" width="{size}" height="{size}" viewBox="0 0 {size} {size}" style="color:{color}">
        <circle class="ring-bg" cx="{size/2}" cy="{size/2}" r="{radius}" stroke-width="{stroke}" />
        <circle class="ring-fg" cx="{size/2}" cy="{size/2}" r="{radius}" stroke-width="{stroke}"
                stroke-dasharray="{circumference:.2f}"
                style="--ring-full:{circumference:.2f}; --ring-offset:{offset:.2f}; stroke-dashoffset:{circumference:.2f}; transform:rotate(-90deg); transform-origin:50% 50%;" />
        <text x="50%" y="53%" text-anchor="middle" font-size="18">{pct*100:.0f}%</text>
    </svg>
    </div>
    """, unsafe_allow_html=True)


def animated_words(text, base_delay=0.05, css_class="word"):
    """Wraps each word in a span with a staggered fade-in delay for a word-by-word reveal."""
    words = text.split(" ")
    spans = "".join(
        f'<span class="{css_class}" style="animation-delay:{i*base_delay:.2f}s">{w}&nbsp;</span>'
        for i, w in enumerate(words)
    )
    return spans

# ============================================================
# MODEL LOADING (with graceful synthetic fallback demo model)
# ============================================================
@st.cache_resource
def load_model():
    model_path = MODELS / "customer_segment_classifier.joblib"
    if model_path.exists():
        return joblib.load(model_path), True

    # --- Synthetic fallback so the dashboard is fully usable without a real model file ---
    rng = np.random.default_rng(42)
    n = 2000
    synth = pd.concat([
        pd.DataFrame({
            "Recency": rng.integers(0, 20, n // 4),
            "Frequency": rng.integers(10, 60, n // 4),
            "Monetary": rng.uniform(3000, 20000, n // 4),
            "Segment": "Champions",
        }),
        pd.DataFrame({
            "Recency": rng.integers(10, 60, n // 4),
            "Frequency": rng.integers(4, 20, n // 4),
            "Monetary": rng.uniform(500, 4000, n // 4),
            "Segment": "Loyal Customers",
        }),
        pd.DataFrame({
            "Recency": rng.integers(0, 40, n // 4),
            "Frequency": rng.integers(1, 4, n // 4),
            "Monetary": rng.uniform(20, 800, n // 4),
            "Segment": "New / Occasional",
        }),
        pd.DataFrame({
            "Recency": rng.integers(120, 600, n // 4),
            "Frequency": rng.integers(1, 5, n // 4),
            "Monetary": rng.uniform(10, 500, n // 4),
            "Segment": "Hibernating",
        }),
    ], ignore_index=True)

    X = np.log1p(synth[FEATURES])
    y = synth["Segment"]
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    return model, False


clf, is_real_model = load_model()


@st.cache_data
def reference_stats():
    """Population percentile reference used for radar chart comparisons."""
    rng = np.random.default_rng(7)
    n = 3000
    ref = pd.DataFrame({
        "Recency": rng.gamma(2, 40, n),
        "Frequency": rng.gamma(2, 4, n),
        "Monetary": rng.gamma(2, 400, n),
    })
    return ref


REF = reference_stats()


def sharpen_proba(proba, temperature):
    """Rescales a probability vector for DISPLAY purposes only.
    temperature < 1.0 sharpens toward the top class; temperature == 1.0 leaves it untouched.
    This never changes which class is predicted (argmax is invariant to this transform) —
    it only changes how confident the number looks."""
    if temperature >= 0.999:
        return proba
    eps = 1e-9
    scaled = np.power(np.asarray(proba, dtype=float) + eps, 1.0 / temperature)
    return scaled / scaled.sum(axis=-1, keepdims=True)


def predict_segment(recency, frequency, monetary, temperature=1.0):
    """IMPORTANT: this classifier was trained on log1p-transformed RFM values,
    so raw inputs must go through the same transform before prediction."""
    raw = pd.DataFrame({"Recency": [recency], "Frequency": [frequency], "Monetary": [monetary]})
    transformed = np.log1p(raw[FEATURES])
    pred = clf.predict(transformed)[0]
    proba = clf.predict_proba(transformed)[0]
    proba_display = sharpen_proba(proba, temperature)
    return pred, proba_display


def predict_segment_batch(df, temperature=1.0):
    transformed = np.log1p(df[FEATURES])
    preds = clf.predict(transformed)
    proba = clf.predict_proba(transformed)
    proba_display = sharpen_proba(proba, temperature)
    confidences = proba_display.max(axis=1)
    return preds, confidences


def percentile_of(value, series, invert=False):
    pct = (series < value).mean() * 100
    return 100 - pct if invert else pct


# ============================================================
# UNIVERSAL DATASET LOADING — accepts (almost) any file, any layout
# ============================================================
RFM_SYNONYMS = {
    "Recency": ["recency", "recencydays", "dayssincelastpurchase", "dayssincelastorder",
                "lastpurchasedays", "daysfromlastpurchase", "recencyindays", "rscore"],
    "Frequency": ["frequency", "numorders", "ordercount", "orders", "numpurchases",
                  "purchasecount", "numtransactions", "transactioncount", "totalorders", "fscore"],
    "Monetary": ["monetary", "totalspend", "spend", "totalamount", "revenue", "sales",
                 "totalprice", "amountspent", "lifetimevalue", "ltv", "totalrevenue", "mscore"],
}
CUSTOMER_ID_HINTS = ["customerid", "custid", "clientid", "userid", "buyerid", "accountid", "customer"]
DATE_HINTS = ["invoicedate", "orderdate", "date", "purchasedate", "transactiondate", "createdat", "timestamp"]
AMOUNT_HINTS = ["amount", "totalprice", "revenue", "sales", "price", "unitprice", "spend", "linetotal", "total"]
QTY_HINTS = ["quantity", "qty", "units", "unitssold"]


def _normalize(col):
    return re.sub(r"[^a-z0-9]", "", str(col).lower())


def guess_column(columns, keywords):
    """Best-effort match of a dataset's real column names to a canonical concept (e.g. 'Recency')."""
    norm = {c: _normalize(c) for c in columns}
    for kw in keywords:
        kwn = _normalize(kw)
        for c, nc in norm.items():
            if nc == kwn:
                return c
    for kw in keywords:
        kwn = _normalize(kw)
        for c, nc in norm.items():
            if kwn and (kwn in nc or nc in kwn):
                return c
    return None


def load_any_dataset(uploaded):
    """Reads CSV, TSV, Excel, or JSON into a DataFrame, with encoding fallback for text formats."""
    name = uploaded.name.lower()
    if name.endswith((".xlsx", ".xls")):
        try:
            return pd.read_excel(uploaded)
        except ImportError as e:
            raise RuntimeError(
                "Reading Excel files needs the 'openpyxl' package, which isn't installed on this "
                "deployment. Add `openpyxl` to requirements.txt, or upload this as a CSV instead."
            ) from e
    if name.endswith(".json"):
        return pd.read_json(uploaded)
    sep = "\t" if name.endswith(".tsv") else ","
    try:
        return pd.read_csv(uploaded, sep=sep)
    except UnicodeDecodeError:
        uploaded.seek(0)
        try:
            return pd.read_csv(uploaded, sep=sep, encoding="utf-8-sig")
        except UnicodeDecodeError:
            uploaded.seek(0)
            return pd.read_csv(uploaded, sep=sep, encoding="cp1252")


def compute_rfm_from_transactions(df, customer_col, date_col, amount_col):
    """Aggregates a raw transaction/order log into one Recency/Frequency/Monetary row per customer."""
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col, customer_col])
    work[amount_col] = pd.to_numeric(work[amount_col], errors="coerce").fillna(0)

    snapshot = work[date_col].max() + pd.Timedelta(days=1)
    grouped = work.groupby(customer_col).agg(
        Recency=(date_col, lambda s: (snapshot - s.max()).days),
        Frequency=(date_col, "count"),
        Monetary=(amount_col, "sum"),
    ).reset_index().rename(columns={customer_col: "CustomerID"})
    grouped["Recency"] = grouped["Recency"].clip(lower=0)
    grouped["Monetary"] = grouped["Monetary"].clip(lower=0)
    return grouped


# ============================================================
# SESSION STATE (prediction history log)
# ============================================================
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(
        columns=["Timestamp", "Recency", "Frequency", "Monetary", "Segment", "Confidence"]
    )

# ============================================================
# HERO HEADER
# ============================================================
_hero_sparks = "".join(
    f'<span class="spark" style="top:{y}%; left:{x}%; animation-delay:{d:.1f}s;"></span>'
    for x, y, d in [(8,20,0),(18,65,.4),(30,15,.9),(42,72,.3),(55,25,1.2),
                     (66,60,.6),(78,18,1.5),(88,55,.2),(95,30,1.0),(50,85,.8)]
)
st.markdown(f"""
<div class="hero">
  {_hero_sparks}
  <h1>🛍️ Customer Segment Predictor</h1>
  <p class="subtitle">{animated_words("RFM-based classifier for customer segmentation, scoring, and retention strategy.")}</p>
</div>
""", unsafe_allow_html=True)


if not is_real_model:
    st.warning(
        "⚠️ No trained model found at `models/customer_segment_classifier.joblib` — "
        "running on a **synthetic demo model** so you can explore the app. Drop your real "
        "`.joblib` file in the `models/` folder to use live predictions.",
        icon="⚠️",
    )

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("📊 Model Info")
    kpi_card("Model type", type(clf).__name__, delay=0.0)
    st.write("")
    kpi_card("Status", "✅ Live model" if is_real_model else "🧪 Demo model", delay=0.1)
    st.write("")
    kpi_card("Held-out accuracy", "99.19%" if is_real_model else "n/a (synthetic)", delay=0.2)

    st.divider()
    st.header("🔬 Confidence Display")
    conf_temp = st.slider(
        "Sharpening", min_value=0.4, max_value=1.0, value=1.0, step=0.05,
        help=(
            "Adjusts how confidence percentages are DISPLAYED — it never changes the "
            "predicted segment (that's fixed by the model). Lower values sharpen the "
            "top probability so borderline cases read as more decisive. 1.0 = raw, "
            "unmodified model output."
        ),
    )
    if conf_temp < 1.0:
        st.caption(f"⚡ Sharpened at T={conf_temp:.2f} — predictions unchanged, only the % display is amplified.")
    else:
        st.caption("Showing raw model confidence (unmodified).")

    st.divider()
    st.header("🕘 Session History")
    st.caption(f"{len(st.session_state.history)} prediction(s) this session")
    if not st.session_state.history.empty:
        st.dataframe(st.session_state.history.tail(5), use_container_width=True, height=180)
        st.download_button(
            "⬇️ Download session history",
            st.session_state.history.to_csv(index=False).encode("utf-8"),
            "prediction_history.csv",
            "text/csv",
        )
        if st.button("🗑️ Clear history"):
            st.session_state.history = st.session_state.history.iloc[0:0]
            st.rerun()

    st.divider()
    st.header("📖 Segment Legend")
    for i, (seg, info) in enumerate(SEGMENT_INFO.items()):
        color = SEGMENT_COLORS[seg]
        st.markdown(f"""
        <div class="segment-chip" style="background:linear-gradient(135deg,{color}22,{color}08); animation-delay:{i*0.08}s;">
            <h4>{info['emoji']} {seg}</h4>
            <p>{info['desc']}</p>
        </div>
        """, unsafe_allow_html=True)
        st.write("")

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Predict one customer",
    "📄 Score a CSV file",
    "🧠 Model insights",
    "ℹ️ About RFM",
])

# ------------------------------------------------------------
# TAB 1 — SINGLE PREDICTION
# ------------------------------------------------------------
with tab1:
    st.subheader("Enter customer details")

    input_mode = st.radio("Input method", ["Sliders", "Number fields"], horizontal=True)
    col1, col2, col3 = st.columns(3)

    if input_mode == "Sliders":
        with col1:
            recency = st.slider("Recency (days since last purchase)", 0, 400, 30)
        with col2:
            frequency = st.slider("Frequency (number of orders)", 1, 100, 3)
        with col3:
            monetary = st.slider("Monetary (total spend, £)", 0.0, 20000.0, 500.0, step=10.0)
    else:
        with col1:
            recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=1000, value=30)
        with col2:
            frequency = st.number_input("Frequency (number of orders)", min_value=1, max_value=500, value=3)
        with col3:
            monetary = st.number_input("Monetary (total spend, £)", min_value=0.0, max_value=1_000_000.0, value=500.0, step=10.0)

    predict_clicked = st.button("Predict Segment", type="primary", use_container_width=False)

    if predict_clicked:
        segment, proba = predict_segment(recency, frequency, monetary, temperature=conf_temp)
        info = SEGMENT_INFO[segment]
        confidence = proba.max()
        sorted_proba = np.sort(proba)[::-1]
        margin = sorted_proba[0] - sorted_proba[1] if len(sorted_proba) > 1 else 1.0

        # log to session history
        new_row = pd.DataFrame([{
            "Timestamp": datetime.now().strftime("%H:%M:%S"),
            "Recency": recency, "Frequency": frequency, "Monetary": monetary,
            "Segment": segment, "Confidence": round(float(confidence), 3),
        }])
        st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True)

        if confidence >= 0.95:
            confetti_burst(220)
            st.balloons()
            st.toast(f"Outstanding match: {segment}! 🌟", icon="🎊")
        elif confidence >= 0.85:
            confetti_burst(140, colors=[SEGMENT_COLORS[segment], "#ffffff"])
            st.toast(f"High-confidence match: {segment}! 🎉", icon="✨")
        else:
            st.toast("Prediction logged to session history.", icon="📝")

        res_col, radar_col = st.columns([1, 1])

        with res_col:
            color = SEGMENT_COLORS[segment]
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown(f"""
                <div class="result-card" style="background:linear-gradient(135deg,{color},{color}cc); flex:1;">
                    <div>
                        <span class="result-badge">{info['priority']} priority</span>
                        <div class="result-title">{info['emoji']} {segment}</div>
                        <div class="result-desc">{info['desc']}</div>
                        <div class="result-desc" style="margin-top:.5rem;"><b>Recommended action:</b> {info['action']}</div>
                        <div class="confidence-track">
                            <div class="confidence-fill" style="width:{confidence*100:.1f}%;"></div>
                        </div>
                        <div class="confidence-label">{confidence:.1%} confidence</div>
                        <div class="confidence-label">margin over runner-up: {margin:.1%}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.write("")
                confidence_ring(confidence, color)
            st.write("")

            prob_df = pd.DataFrame({"Segment": clf.classes_, "Probability": proba}).sort_values("Probability", ascending=True)
            fig = px.bar(prob_df, x="Probability", y="Segment", orientation="h", text="Probability",
                         color="Segment", color_discrete_map=SEGMENT_COLORS)
            fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
            fig.update_layout(showlegend=False, xaxis=dict(range=[0, 1], tickformat=".0%"),
                               yaxis_title="", height=280, margin=dict(t=10, b=10),
                               transition_duration=500)
            st.plotly_chart(fig, use_container_width=True)

        with radar_col:
            st.markdown("**How this customer compares to the population**")
            r_pct = percentile_of(recency, REF["Recency"], invert=True)  # lower recency = better
            f_pct = percentile_of(frequency, REF["Frequency"])
            m_pct = percentile_of(monetary, REF["Monetary"])

            radar = go.Figure()
            radar.add_trace(go.Scatterpolar(
                r=[r_pct, f_pct, m_pct, r_pct],
                theta=["Recency (recentness)", "Frequency", "Monetary", "Recency (recentness)"],
                fill="toself", name="This customer",
                line_color=SEGMENT_COLORS[segment],
            ))
            radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100], ticksuffix="%")),
                showlegend=False, height=350, margin=dict(t=20, b=20),
                transition_duration=600,
            )
            st.plotly_chart(radar, use_container_width=True)
            st.caption("Percentile rank vs. a reference customer population (higher = stronger on that dimension).")

    if not st.session_state.history.empty:
        with st.expander(f"📜 Full session history ({len(st.session_state.history)} predictions)"):
            st.dataframe(st.session_state.history, use_container_width=True)
            trend = px.bar(st.session_state.history["Segment"].value_counts().reset_index(),
                           x="Segment", y="count", color="Segment", color_discrete_map=SEGMENT_COLORS)
            trend.update_layout(showlegend=False, height=250, margin=dict(t=10, b=10))
            st.plotly_chart(trend, use_container_width=True)

# ------------------------------------------------------------
# TAB 2 — BATCH CSV SCORING
# ------------------------------------------------------------
with tab2:
    st.subheader("Upload any customer dataset")
    st.caption(
        "CSV, TSV, Excel, or JSON. Already have Recency/Frequency/Monetary columns? "
        "We'll auto-detect them. Have raw order/transaction data instead? We'll build "
        "RFM for you automatically."
    )
    uploaded = st.file_uploader("Choose a file", type=["csv", "tsv", "xlsx", "xls", "json"])

    if uploaded is not None:
        try:
            raw_data = load_any_dataset(uploaded)
        except Exception as e:
            st.error(f"Couldn't read this file: {e}")
            st.stop()

        if raw_data.empty or raw_data.shape[1] == 0:
            st.error("This file loaded but appears to be empty.")
            st.stop()

        st.success(f"Loaded **{uploaded.name}** — {raw_data.shape[0]:,} rows × {raw_data.shape[1]} columns")
        with st.expander("🔍 Preview raw data", expanded=False):
            st.dataframe(raw_data.head(20), use_container_width=True)

        cols = list(raw_data.columns)
        guess_r = guess_column(cols, RFM_SYNONYMS["Recency"])
        guess_f = guess_column(cols, RFM_SYNONYMS["Frequency"])
        guess_m = guess_column(cols, RFM_SYNONYMS["Monetary"])
        guess_cust = guess_column(cols, CUSTOMER_ID_HINTS)
        guess_date = guess_column(cols, DATE_HINTS)
        guess_amount = guess_column(cols, AMOUNT_HINTS)
        guess_qty = guess_column(cols, QTY_HINTS)

        has_rfm_guess = guess_r and guess_f and guess_m
        looks_transactional = guess_cust and guess_date and not has_rfm_guess

        st.markdown("**How is this data organized?**")
        mode = st.radio(
            "Data layout",
            ["Already has Recency / Frequency / Monetary columns",
             "Raw order or transaction log (one row per purchase)"],
            index=1 if looks_transactional else 0,
            label_visibility="collapsed",
        )

        data = None

        if mode.startswith("Already"):
            if has_rfm_guess:
                st.caption(f"Auto-detected — Recency: `{guess_r}` · Frequency: `{guess_f}` · Monetary: `{guess_m}`. Adjust below if wrong.")
            c1, c2, c3 = st.columns(3)
            with c1:
                col_r = st.selectbox("Recency column", cols, index=cols.index(guess_r) if guess_r in cols else 0)
            with c2:
                col_f = st.selectbox("Frequency column", cols, index=cols.index(guess_f) if guess_f in cols else 0)
            with c3:
                col_m = st.selectbox("Monetary column", cols, index=cols.index(guess_m) if guess_m in cols else 0)

            if len({col_r, col_f, col_m}) < 3:
                st.error("Recency, Frequency, and Monetary need to be three different columns.")
                st.stop()

            data = raw_data.rename(columns={col_r: "Recency", col_f: "Frequency", col_m: "Monetary"}).copy()
            for c in ["Recency", "Frequency", "Monetary"]:
                data[c] = pd.to_numeric(data[c], errors="coerce")
            bad = data[["Recency", "Frequency", "Monetary"]].isna().any(axis=1).sum()
            if bad:
                st.warning(f"{bad} row(s) had non-numeric Recency/Frequency/Monetary values and will be dropped.")
                data = data.dropna(subset=["Recency", "Frequency", "Monetary"])

        else:
            st.caption("Map the columns below — we'll calculate Recency, Frequency, and Monetary per customer.")
            c1, c2 = st.columns(2)
            with c1:
                col_cust = st.selectbox("Customer ID column", cols, index=cols.index(guess_cust) if guess_cust in cols else 0)
            with c2:
                col_date = st.selectbox("Order / purchase date column", cols, index=cols.index(guess_date) if guess_date in cols else 0)

            amount_mode = st.radio("How is spend represented?", ["Single amount column", "Quantity × Unit price"], horizontal=True)
            if amount_mode == "Single amount column":
                col_amount = st.selectbox("Amount column", cols, index=cols.index(guess_amount) if guess_amount in cols else 0)
            else:
                qc1, qc2 = st.columns(2)
                with qc1:
                    col_qty = st.selectbox("Quantity column", cols, index=cols.index(guess_qty) if guess_qty in cols else 0)
                with qc2:
                    col_price = st.selectbox("Unit price column", cols, index=cols.index(guess_amount) if guess_amount in cols else 0)
                raw_data = raw_data.copy()
                raw_data["_computed_amount"] = pd.to_numeric(raw_data[col_qty], errors="coerce") * pd.to_numeric(raw_data[col_price], errors="coerce")
                col_amount = "_computed_amount"

            if st.button("⚙️ Compute RFM from this data", type="primary"):
                with st.spinner("Aggregating transactions into Recency / Frequency / Monetary..."):
                    try:
                        data = compute_rfm_from_transactions(raw_data, col_cust, col_date, col_amount)
                    except Exception as e:
                        st.error(f"Couldn't compute RFM: {e}")
                        st.stop()
                st.session_state["_computed_rfm"] = data
                st.success(f"Computed RFM for {len(data):,} customers from {raw_data.shape[0]:,} transaction rows.")

            data = st.session_state.get("_computed_rfm") if data is None else data

        if data is not None and len(data) > 0:
            with st.spinner("Scoring customers..."):
                preds, confidences = predict_segment_batch(data, temperature=conf_temp)
                data["Predicted_Segment"] = preds
                data["Confidence"] = (confidences * 100).round(1)

            # KPI row (animated cards, staggered entrance)
            top_seg = data["Predicted_Segment"].value_counts().idxmax()
            low_conf = (confidences < 0.6).sum()
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                kpi_card("Customers scored", f"{len(data):,}", delay=0.0)
            with k2:
                kpi_card("Avg. confidence", f"{confidences.mean():.1%}", delay=0.1)
            with k3:
                kpi_card("Largest segment", top_seg, delay=0.2)
            with k4:
                kpi_card("Low-confidence cases", f"{low_conf}", delay=0.3)
            st.caption("Low-confidence = predictions under 60% probability — worth a manual look.")

            st.divider()

            filt_col, sort_col = st.columns([2, 1])
            with filt_col:
                seg_filter = st.multiselect("Filter by segment", options=sorted(data["Predicted_Segment"].unique()),
                                             default=sorted(data["Predicted_Segment"].unique()))
            with sort_col:
                sort_by_conf = st.checkbox("Sort by confidence (lowest first)")

            view = data[data["Predicted_Segment"].isin(seg_filter)]
            if sort_by_conf:
                view = view.sort_values("Confidence")

            st.write(f"Showing {len(view)} of {len(data)} scored customers:")
            st.dataframe(
                view,
                use_container_width=True,
                column_config={
                    "Confidence": st.column_config.ProgressColumn(
                        "Confidence", min_value=0, max_value=100, format="%.0f%%"
                    )
                },
            )

            c1, c2 = st.columns(2)
            with c1:
                counts = data["Predicted_Segment"].value_counts().reset_index()
                counts.columns = ["Segment", "Count"]
                fig2 = px.pie(counts, names="Segment", values="Count", hole=0.5,
                              color="Segment", color_discrete_map=SEGMENT_COLORS, title="Segment distribution")
                st.plotly_chart(fig2, use_container_width=True)
            with c2:
                fig3 = px.scatter(
                    data, x="Recency", y="Monetary", size="Frequency", color="Predicted_Segment",
                    color_discrete_map=SEGMENT_COLORS, hover_data=["Frequency", "Confidence"],
                    title="Recency vs. Monetary (bubble size = Frequency)",
                )
                st.plotly_chart(fig3, use_container_width=True)

            csv_out = data.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download scored CSV", csv_out, "scored_customers.csv", "text/csv", type="primary")
    else:
        st.session_state.pop("_computed_rfm", None)
        st.info("Need a template? Download one below, fill it in, and upload it back — or just upload your own raw order data instead.")
        template = pd.DataFrame({"Recency": [5, 60, 200], "Frequency": [15, 3, 1], "Monetary": [7000, 600, 80]})
        st.dataframe(template, use_container_width=True)
        st.download_button("⬇️ Download template CSV", template.to_csv(index=False).encode("utf-8"), "template.csv", "text/csv")

# ------------------------------------------------------------
# TAB 3 — MODEL INSIGHTS
# ------------------------------------------------------------
with tab3:
    st.subheader("Model insights")

    if hasattr(clf, "feature_importances_"):
        imp = pd.DataFrame({"Feature": FEATURES, "Importance": clf.feature_importances_}).sort_values("Importance")
        fig_imp = px.bar(imp, x="Importance", y="Feature", orientation="h", title="Feature importance")
        fig_imp.update_layout(height=280, margin=dict(t=40, b=10))
        st.plotly_chart(fig_imp, use_container_width=True)
    elif hasattr(clf, "coef_"):
        coefs = pd.DataFrame(clf.coef_, columns=FEATURES, index=clf.classes_)
        st.write("Logistic regression coefficients (log-odds impact per class):")
        st.dataframe(coefs, use_container_width=True)
        fig_coef = px.imshow(coefs, color_continuous_scale="RdBu", aspect="auto",
                              text_auto=".2f", title="Coefficient heatmap")
        fig_coef.update_layout(height=280, margin=dict(t=40, b=10))
        st.plotly_chart(fig_coef, use_container_width=True)
    else:
        st.info("This model type doesn't expose a standard feature-importance attribute.")

    st.divider()
    st.markdown("**Decision space preview** — synthetic grid colored by predicted segment "
                "(Frequency fixed at its median value).")
    med_freq = REF["Frequency"].median()
    grid_r = np.linspace(0, REF["Recency"].quantile(0.98), 40)
    grid_m = np.linspace(0, REF["Monetary"].quantile(0.98), 40)
    gr, gm = np.meshgrid(grid_r, grid_m)
    grid_df = pd.DataFrame({"Recency": gr.ravel(), "Frequency": med_freq, "Monetary": gm.ravel()})
    grid_preds = clf.predict(np.log1p(grid_df[FEATURES]))
    grid_df["Segment"] = grid_preds

    fig_grid = px.scatter(
        grid_df, x="Recency", y="Monetary", color="Segment", color_discrete_map=SEGMENT_COLORS,
        opacity=0.6, height=420,
    )
    fig_grid.update_traces(marker=dict(size=9, symbol="square"))
    st.plotly_chart(fig_grid, use_container_width=True)
    st.caption(f"Frequency held constant at {med_freq:.1f} orders (population median) to visualize the 2D decision surface.")

# ------------------------------------------------------------
# TAB 4 — ABOUT / METHODOLOGY
# ------------------------------------------------------------
with tab4:
    st.subheader("What is RFM segmentation?")
    st.markdown("""
RFM stands for **Recency, Frequency, Monetary** — three simple but powerful signals
derived from raw transaction history:

- **Recency** — how many days since the customer's last purchase (lower is better).
- **Frequency** — how many orders they've placed in the observation window (higher is better).
- **Monetary** — how much they've spent in total (higher is better).

Because these values are typically right-skewed (a few customers spend enormously more
than the rest), the model applies a **log1p transform** before scoring, which is why
this app always normalizes raw inputs the same way before calling `predict`.
""")

    cols = st.columns(4)
    for i, (c, (seg, info)) in enumerate(zip(cols, SEGMENT_INFO.items())):
        color = SEGMENT_COLORS[seg]
        with c:
            st.markdown(f"""
            <div class="segment-chip" style="background:linear-gradient(160deg,{color}26,{color}0a); min-height:190px; animation-delay:{i*0.1}s;">
                <h4>{info['emoji']} {seg}</h4>
                <p>{info['desc']}</p>
                <p style="margin-top:.5rem;"><b>Action:</b> {info['action']}</p>
                <p><b>Priority:</b> {info['priority']}</p>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
**Using this dashboard**
1. Predict a single customer's segment interactively, or
2. Score a whole CSV of customers at once, or
3. Inspect what's driving the model's decisions in *Model insights*.

Replace the demo model with your own by placing a trained classifier at
`models/customer_segment_classifier.joblib`. It must implement `.predict()` and
`.predict_proba()` and expect `log1p`-transformed `Recency`, `Frequency`, `Monetary` features.
""")

st.markdown("---")
st.caption(
    ("Model: " + type(clf).__name__ if is_real_model else "Model: synthetic demo (RandomForest)")
    + " · Trained on RFM features from e-commerce transaction data · "
    + ("Validated on a fully held-out 20% of customers" if is_real_model else "Not validated — demo only")
)
