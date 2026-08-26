from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Customer Segment Predictor", page_icon="🛍️", layout="wide")

MODELS = Path(__file__).parent / "models"
FEATURES = ["Recency", "Frequency", "Monetary"]

SEGMENT_COLORS = {
    "Champions": "#00CEC9",
    "Loyal Customers": "#6C5CE7",
    "New / Occasional": "#FDCB6E",
    "Hibernating": "#8B93A7",
}
SEGMENT_INFO = {
    "Champions": {"emoji": "🏆", "desc": "Recent, frequent, high spend — your most valuable customers.",
                  "action": "Reward with loyalty perks and early access to new products."},
    "Loyal Customers": {"emoji": "💙", "desc": "Solid frequency and spend, moderately recent.",
                         "action": "Good candidates for upsell/cross-sell and bundle offers."},
    "New / Occasional": {"emoji": "🌱", "desc": "Bought recently but with few orders so far.",
                          "action": "Target with second-purchase incentives to build the habit."},
    "Hibernating": {"emoji": "😴", "desc": "Long time since last purchase, low historical spend.",
                     "action": "Low-cost reactivation campaigns only."},
}

# ============================================================
# LOAD MODEL (cached so it only loads once)
# ============================================================
@st.cache_resource
def load_model():
    return joblib.load(MODELS / "customer_segment_classifier.joblib")

clf = load_model()


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
    return clf.predict(transformed)


# ============================================================
# UI
# ============================================================
st.title("🛍️ Customer Segment Predictor")
st.caption("Random Forest / Logistic Regression classifier trained on RFM features — 99.19% accuracy on held-out customers.")

tab1, tab2 = st.tabs(["🔮 Predict one customer", "📄 Score a CSV file"])

with tab1:
    st.subheader("Enter customer details")
    col1, col2, col3 = st.columns(3)
    with col1:
        recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=1000, value=30)
    with col2:
        frequency = st.number_input("Frequency (number of orders)", min_value=1, max_value=500, value=3)
    with col3:
        monetary = st.number_input("Monetary (total spend, £)", min_value=0.0, max_value=1_000_000.0, value=500.0, step=10.0)

    if st.button("Predict Segment", type="primary"):
        segment, proba = predict_segment(recency, frequency, monetary)
        info = SEGMENT_INFO[segment]

        st.success(f"### {info['emoji']} Predicted Segment: **{segment}**")
        st.write(info["desc"])
        st.info(f"**Recommended action:** {info['action']}")

        prob_df = pd.DataFrame({"Segment": clf.classes_, "Probability": proba}).sort_values("Probability", ascending=True)
        fig = px.bar(prob_df, x="Probability", y="Segment", orientation="h", text="Probability",
                     color="Segment", color_discrete_map=SEGMENT_COLORS)
        fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
        fig.update_layout(showlegend=False, xaxis=dict(range=[0, 1], tickformat=".0%"), yaxis_title="", height=300)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Upload a CSV with columns: Recency, Frequency, Monetary")
    uploaded = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded is not None:
        data = pd.read_csv(uploaded)
        required = {"Recency", "Frequency", "Monetary"}
        if not required.issubset(data.columns):
            st.error(f"CSV must contain columns: {sorted(required)}")
        else:
            data["Predicted_Segment"] = predict_segment_batch(data)
            st.write(f"Scored {len(data)} customers:")
            st.dataframe(data, use_container_width=True)

            counts = data["Predicted_Segment"].value_counts().reset_index()
            counts.columns = ["Segment", "Count"]
            fig2 = px.pie(counts, names="Segment", values="Count", hole=0.5,
                          color="Segment", color_discrete_map=SEGMENT_COLORS)
            st.plotly_chart(fig2, use_container_width=True)

            csv_out = data.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download scored CSV", csv_out, "scored_customers.csv", "text/csv")
    else:
        st.info("Need a template? Download one below, fill it in, and upload it back.")
        template = pd.DataFrame({"Recency": [5, 60, 200], "Frequency": [15, 3, 1], "Monetary": [7000, 600, 80]})
        st.dataframe(template, use_container_width=True)
        st.download_button("⬇️ Download template CSV", template.to_csv(index=False).encode("utf-8"), "template.csv", "text/csv")

st.markdown("---")
st.caption("Model: Logistic Regression · Trained on RFM features from e-commerce transaction data · Validated on a fully held-out 20% of customers")
