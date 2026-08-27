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
.hero p  { margin: .45rem 0 0 0; opacity: .95; animation: fadeInUp .85s ease both; position: relative; z-index: 1; }

/* ---------- Reusable entrance animation ---------- */
.animate-in { animation: fadeInUp .55s cubic-bezier(.2,.8,.2,1) both; }

/* ---------- KPI / metric cards ---------- */
.kpi-card {
    background: linear-gradient(145deg, #ffffff12, #ffffff05);
    border: 1px solid #ffffff22; border-radius: 14px; padding: 1rem 1rem;
    text-align: center; transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
    animation: fadeInUp .5s ease both;
}
.kpi-card:hover { transform: translateY(-5px); box-shadow: 0 12px 26px rgba(0,0,0,.18); border-color: #ffffff44; }
.kpi-card .kpi-label { font-size: .78rem; opacity: .7; letter-spacing: .02em; text-transform: uppercase; }
.kpi-card .kpi-value { font-size: 1.7rem; font-weight: 700; margin-top: .15rem; }

/* ---------- Prediction result card ---------- */
.result-card {
    border-radius: 18px; padding: 1.5rem 1.7rem; color: white; position: relative;
    overflow: hidden; animation: popIn .55s cubic-bezier(.2,.85,.3,1.2) both;
    box-shadow: 0 14px 34px rgba(0,0,0,.22);
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

/* ---------- Segment legend chips ---------- */
.segment-chip {
    border-radius: 14px; padding: .9rem 1rem; text-align: left;
    transition: transform .2s ease, box-shadow .2s ease; animation: fadeInUp .5s ease both;
    border: 1px solid #ffffff1f;
}
.segment-chip:hover { transform: translateY(-4px) scale(1.015); box-shadow: 0 10px 22px rgba(0,0,0,.16); }
.segment-chip h4 { margin: 0 0 .25rem 0; }
.segment-chip p { margin: 0; font-size: .84rem; opacity: .92; }

/* ---------- Interactive widgets ---------- */
.stButton > button {
    transition: transform .18s ease, box-shadow .18s ease !important;
    border-radius: 10px !important;
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 18px rgba(108,92,231,.35); }
.stButton > button:active { transform: translateY(0) scale(.96); }

[data-baseweb="tab-list"] button { transition: transform .18s ease, color .18s ease; }
[data-baseweb="tab-list"] button:hover { transform: translateY(-2px); }

div[data-testid="stMetricValue"] { animation: fadeInUp .5s ease both; }
div[data-testid="stExpander"] { transition: box-shadow .2s ease; }
img, .stPlotlyChart { animation: fadeInUp .6s ease both; }
</style>
""", unsafe_allow_html=True)


def confetti_burst(pieces=140):
    """Fires a lightweight client-side confetti animation (canvas-confetti via CDN)."""
    components.html(f"""
        <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.9.2/dist/confetti.browser.min.js"></script>
        <script>
        try {{
          confetti({{ particleCount: {pieces}, spread: 80, origin: {{ y: 0.4 }} }});
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


def predict_segment(recency, frequency, monetary):
    """IMPORTANT: this classifier was trained on log1p-transformed RFM values,
    so raw inputs must go through the same transform before prediction."""
    raw = pd.DataFrame({"Recency": [recency], "Frequency": [frequency], "Monetary": [monetary]})
    transformed = np.log1p(raw[FEATURES])
    pred = clf.predict(transformed)[0]
    proba = clf.predict_proba(transformed)[0]
    return pred, proba


def predict_segment_batch(df):
    transformed = np.log1p(df[FEATURES])
    preds = clf.predict(transformed)
    proba = clf.predict_proba(transformed)
    confidences = proba.max(axis=1)
    return preds, confidences


def percentile_of(value, series, invert=False):
    pct = (series < value).mean() * 100
    return 100 - pct if invert else pct


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
st.markdown("""
<div class="hero">
  <h1>🛍️ Customer Segment Predictor</h1>
  <p>RFM-based classifier for customer segmentation, scoring, and retention strategy.</p>
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
        segment, proba = predict_segment(recency, frequency, monetary)
        info = SEGMENT_INFO[segment]
        confidence = proba.max()

        # log to session history
        new_row = pd.DataFrame([{
            "Timestamp": datetime.now().strftime("%H:%M:%S"),
            "Recency": recency, "Frequency": frequency, "Monetary": monetary,
            "Segment": segment, "Confidence": round(float(confidence), 3),
        }])
        st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True)

        if confidence >= 0.85:
            confetti_burst()
            st.toast(f"High-confidence match: {segment}! 🎉", icon="✨")
        else:
            st.toast("Prediction logged to session history.", icon="📝")

        res_col, radar_col = st.columns([1, 1])

        with res_col:
            color = SEGMENT_COLORS[segment]
            st.markdown(f"""
            <div class="result-card" style="background:linear-gradient(135deg,{color},{color}cc);">
                <span class="result-badge">{info['priority']} priority</span>
                <div class="result-title">{info['emoji']} {segment}</div>
                <div class="result-desc">{info['desc']}</div>
                <div class="result-desc" style="margin-top:.5rem;"><b>Recommended action:</b> {info['action']}</div>
                <div class="confidence-track">
                    <div class="confidence-fill" style="width:{confidence*100:.1f}%;"></div>
                </div>
                <div class="confidence-label">{confidence:.1%} confidence</div>
            </div>
            """, unsafe_allow_html=True)
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
    st.subheader("Upload a CSV with columns: Recency, Frequency, Monetary")
    uploaded = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded is not None:
        data = pd.read_csv(uploaded)
        required = {"Recency", "Frequency", "Monetary"}
        if not required.issubset(data.columns):
            st.error(f"CSV must contain columns: {sorted(required)}")
        else:
            with st.spinner("Scoring customers..."):
                preds, confidences = predict_segment_batch(data)
                data["Predicted_Segment"] = preds
                data["Confidence"] = confidences.round(3)

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
                        "Confidence", min_value=0, max_value=1, format="%.0f%%"
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
        st.info("Need a template? Download one below, fill it in, and upload it back.")
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
