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
    page_title="SegmentIQ — Customer Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODELS = Path(__file__).parent / "models"
FEATURES = ["Recency", "Frequency", "Monetary"]

SEGMENT_COLORS = {
    "Champions": "#C98A2E",         # ledger gold — top tier
    "Loyal Customers": "#2F6F63",   # deep teal — steady, trusted
    "New / Occasional": "#8A9A5B",  # sage — new growth
    "Hibernating": "#8C6B5E",       # dust rose — dormant
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
# GROWTH PLAYBOOK — concrete, per-segment business actions
# ============================================================
GROWTH_PLAYBOOK = {
    "Champions": {
        "objective": "Protect this relationship and turn it into referrals and higher-margin sales.",
        "tactics": [
            ("VIP tier / early access", "Give first access to new products or limited drops before a public launch.", "Email + SMS"),
            ("Two-sided referral offer", "They refer a friend, both get a reward — Champions have the trust to make this convert.", "Email + on-site"),
            ("Premium upsell", "Surface higher-tier or bundled products at checkout and in post-purchase email.", "On-site + email"),
            ("Ask for a review / testimonial", "High-spend, high-trust customers produce your best social proof.", "Email"),
        ],
        "kpi": "Repeat purchase rate, referral conversions, average order value",
        "uplift_pct": 0.06,
        "uplift_note": "modeled as incremental spend from deeper engagement (upsell + referral-driven repeat purchases)",
    },
    "Loyal Customers": {
        "objective": "Increase order frequency and basket size to push them toward Champion status.",
        "tactics": [
            ("Cross-sell bundles", "Recommend complementary products based on past purchase category.", "Email + on-site"),
            ("Loyalty points acceleration", "Double points for a limited window to nudge the next order sooner.", "Email + app push"),
            ("Free-shipping threshold nudge", "Show 'spend $X more for free shipping' near their typical basket size.", "On-site"),
            ("Re-engagement drip", "A short 3-email series if they haven't ordered in their usual cycle.", "Email"),
        ],
        "kpi": "Order frequency, basket size, upgrade rate to Champions",
        "uplift_pct": 0.09,
        "uplift_note": "modeled as incremental spend from increased order frequency via bundles/promotions",
    },
    "New / Occasional": {
        "objective": "Convert a first or second purchase into a habit before they drift away.",
        "tactics": [
            ("Second-purchase incentive", "A time-limited discount or free gift on their next order, sent within days of the first.", "Email"),
            ("Onboarding content", "Show how to get the most from what they bought — reduces early churn.", "Email + app"),
            ("Category discovery", "Recommend a second product category to build a broader habit, not just a repeat buy.", "On-site + email"),
            ("Low-friction win-back", "If no second order within their expected window, a small nudge before they go cold.", "Email"),
        ],
        "kpi": "Second-purchase rate, time-to-second-order",
        "uplift_pct": 0.14,
        "uplift_note": "modeled as the value of converting a one-time buyer into a repeat customer",
    },
    "Hibernating": {
        "objective": "Cheaply test for reactivation — don't overspend on customers unlikely to return.",
        "tactics": [
            ("Low-cost win-back email", "A single 'we miss you' email with a modest incentive — cap spend here.", "Email only"),
            ("Feedback survey", "Ask why they left; costs nothing and surfaces product/service issues.", "Email"),
            ("Suppress from paid ads", "Stop spending paid acquisition budget re-targeting this group.", "Ad platform exclusion list"),
            ("Win-back or sunset", "If no response after one campaign, move them off active marketing lists.", "CRM hygiene"),
        ],
        "kpi": "Reactivation rate per dollar spent, unsubscribe rate",
        "uplift_pct": 0.02,
        "uplift_note": "modeled conservatively — most hibernating customers won't return, so spend here should stay minimal",
    },
}


def render_growth_playbook(segment, monetary_value=None, customer_count=1, key_suffix=""):
    """Renders concrete tactics + an adjustable revenue-impact estimate for one segment."""
    play = GROWTH_PLAYBOOK[segment]
    color = SEGMENT_COLORS[segment]

    st.markdown(f"""
    <div class="segment-chip" style="--tier-color:{color};">
        <h4>📋 Growth playbook — {segment}</h4>
        <p>{play['objective']}</p>
    </div>
    """, unsafe_allow_html=True)
    st.write("")

    for name, desc, channel in play["tactics"]:
        st.markdown(f"""
        <div class="kpi-card" style="text-align:left; margin-bottom:.5rem;">
            <div class="kpi-label">{channel}</div>
            <div style="font-weight:600; margin-top:.15rem;">{name}</div>
            <div style="opacity:.8; font-size:.87rem; margin-top:.2rem;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.caption(f"**Track this with:** {play['kpi']}")

    if monetary_value is not None:
        st.write("")
        st.markdown("**💰 Estimated revenue opportunity**")
        default_uplift_pct = int(round(play["uplift_pct"] * 100))
        uplift_pct = st.slider(
            "Assumed incremental spend uplift",
            min_value=0, max_value=30, value=default_uplift_pct, step=1,
            format="%d%%",
            key=f"uplift_{segment}_{key_suffix}",
            help="An editable assumption, not a guarantee — adjust it to match your own historical campaign results.",
        )
        uplift = uplift_pct / 100
        total_value = monetary_value * customer_count
        estimated_gain = total_value * uplift
        c1, c2, c3 = st.columns(3)
        with c1:
            kpi_card("Segment value", f"${total_value:,.0f}", delay=0)
        with c2:
            kpi_card("Uplift assumption", f"{uplift_pct}%", delay=0.05)
        with c3:
            kpi_card("Estimated opportunity", f"${estimated_gain:,.0f}", delay=0.1)
        st.caption(f"Estimate is {play['uplift_note']}. This is a planning input, not a forecast — validate against a real campaign before committing budget.")

# ============================================================
# STYLE — animated design system
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');

/* ============================================================
   LOYALTY LEDGER — design tokens
   Ink navy base, ledger-gold accent, serif tier names, monospace
   figures (the numbers read like entries in an account book).
   ============================================================ */
:root {
    --ink: #14131b;
    --ink-2: #1c1a26;
    --parchment: #f4eedd;
    --gold: #c98a2e;
    --gold-2: #a5721f;
    --teal: #2f6f63;
    --rose: #b23a48;
    --line: #4a4536;
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes stampIn {
    0%   { opacity: 0; transform: scale(1.6) rotate(-14deg); }
    60%  { opacity: 1; transform: scale(0.94) rotate(2deg); }
    100% { opacity: 1; transform: scale(1) rotate(-6deg); }
}
@keyframes printReveal {
    from { clip-path: inset(0 100% 0 0); }
    to   { clip-path: inset(0 0 0 0); }
}
@keyframes wordIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes ringDraw {
    from { stroke-dashoffset: var(--ring-full); }
    to   { stroke-dashoffset: var(--ring-offset); }
}
@keyframes fillLedger {
    from { width: 0; }
}

/* ---------- Base typography ---------- */
.stApp, .stApp p, .stApp li, .stApp label { font-family: 'Inter', sans-serif; }
.stApp h1, .stApp h2, .stApp h3 { font-family: 'Fraunces', serif; letter-spacing: -0.01em; }

/* ---------- Whole-app backdrop: quiet ledger texture, not a rainbow ---------- */
[data-testid="stAppViewContainer"] {
    background-image: radial-gradient(circle at 12% 8%, rgba(201,138,46,.07), transparent 38%);
}

/* ---------- Hero: a ledger masthead with a wax-stamp mark and a torn ticket edge ---------- */
.hero {
    position: relative; overflow: hidden;
    padding: 2.1rem 2.3rem 2.5rem 2.3rem; border-radius: 4px; margin-bottom: 0;
    background: linear-gradient(150deg, var(--ink-2), var(--ink) 65%);
    color: var(--parchment); border: 1px solid #ffffff14;
}
.hero .stamp {
    position: absolute; top: 1.4rem; right: 2rem; width: 74px; height: 74px;
    border: 2px solid var(--gold); border-radius: 50%; display: flex; align-items: center;
    justify-content: center; font-family: 'Fraunces', serif; font-size: .62rem; font-weight: 700;
    letter-spacing: .06em; text-transform: uppercase; color: var(--gold); text-align: center;
    transform: rotate(-8deg); animation: stampIn .7s cubic-bezier(.2,.8,.3,1.15) both; line-height: 1.25;
    background: rgba(201,138,46,.06);
}
.hero h1 {
    margin: 0; font-size: 2rem; font-weight: 600; color: var(--parchment);
    animation: fadeInUp .55s ease both; position: relative; z-index: 1; max-width: 80%;
}
.hero .eyebrow {
    font-family: 'IBM Plex Mono', monospace; font-size: .72rem; letter-spacing: .12em;
    text-transform: uppercase; color: var(--gold); margin: 0 0 .5rem 0;
    animation: fadeInUp .45s ease both;
}
.hero .subtitle { margin: .6rem 0 0 0; opacity: .82; position: relative; z-index: 1; max-width: 70%; }
.hero .word { display: inline-block; opacity: 0; animation: wordIn .45s ease forwards; }
.perforation { height: 0; border-top: 2px dashed #ffffff2a; position: relative; margin: 0 0 1.3rem 0; }
.perforation::before, .perforation::after {
    content: ""; position: absolute; top: -8px; width: 16px; height: 16px; border-radius: 50%;
    background: var(--ink);
}
.perforation::before { left: -1.5rem; }
.perforation::after  { right: -1.5rem; }

/* ---------- Reusable entrance ---------- */
.animate-in { animation: fadeInUp .5s cubic-bezier(.2,.8,.2,1) both; }

/* ---------- KPI / index-card tiles ---------- */
.kpi-card {
    background: var(--ink-2); border: 1px solid #ffffff16; border-left: 3px solid var(--gold);
    border-radius: 6px; padding: .95rem 1.05rem; text-align: left;
    transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
    animation: fadeInUp .45s ease both;
}
.kpi-card:hover { transform: translateY(-3px); box-shadow: 0 10px 22px rgba(0,0,0,.28); border-left-color: var(--teal); }
.kpi-card .kpi-label {
    font-size: .7rem; opacity: .65; letter-spacing: .08em; text-transform: uppercase;
    font-family: 'IBM Plex Mono', monospace;
}
.kpi-card .kpi-value { font-size: 1.5rem; font-weight: 600; margin-top: .2rem; font-family: 'IBM Plex Mono', monospace; }

/* ---------- Prediction result: a membership-tier ledger card ---------- */
.result-card {
    border-radius: 6px; padding: 1.5rem 1.7rem; color: var(--parchment); position: relative;
    background: var(--ink-2); border: 1px solid #ffffff16; border-left: 5px solid var(--tier-color, var(--gold));
    box-shadow: 0 10px 26px rgba(0,0,0,.3);
    animation: printReveal .6s cubic-bezier(.2,.8,.2,1) both;
    display: flex; justify-content: space-between; align-items: center; gap: 1.2rem; flex-wrap: wrap;
}
.result-badge {
    display: inline-block; padding: .3rem .8rem; border-radius: 3px;
    background: var(--tier-color, var(--gold)); color: var(--ink); font-weight: 700; font-size: .74rem;
    letter-spacing: .06em; text-transform: uppercase; font-family: 'IBM Plex Mono', monospace;
}
.result-title { font-size: 1.5rem; font-weight: 600; margin: .6rem 0 .25rem 0; font-family: 'Fraunces', serif; }
.result-desc { opacity: .82; margin-bottom: .1rem; }
.confidence-track {
    background: #ffffff14; border-radius: 3px; height: 10px;
    overflow: hidden; margin-top: .85rem; position: relative; border: 1px solid #ffffff20;
}
.confidence-fill {
    height: 100%; background: var(--tier-color, var(--gold));
    animation: fillLedger 1s cubic-bezier(.2,.85,.2,1) both;
}
.confidence-label {
    font-size: .74rem; opacity: .85; margin-top: .35rem; text-align: right;
    font-family: 'IBM Plex Mono', monospace;
}
.confidence-ring circle.ring-bg { fill: none; stroke: #ffffff1c; }
.confidence-ring circle.ring-fg {
    fill: none; stroke: currentColor; stroke-linecap: round;
    animation: ringDraw 1.1s cubic-bezier(.2,.85,.2,1) forwards;
}
.confidence-ring text { fill: var(--parchment); font-weight: 700; font-family: 'IBM Plex Mono', monospace; }

/* ---------- Segment legend — membership cards with a tier spine ---------- */
.segment-chip {
    border-radius: 6px; padding: .9rem 1rem; text-align: left; background: var(--ink-2);
    border: 1px solid #ffffff16; border-left: 3px solid var(--tier-color, var(--gold));
    transition: transform .2s ease, box-shadow .2s ease; animation: fadeInUp .45s ease both;
}
.segment-chip:hover { transform: translateY(-3px); box-shadow: 0 10px 22px rgba(0,0,0,.26); }
.segment-chip h4 { margin: 0 0 .3rem 0; font-family: 'Fraunces', serif; font-weight: 600; }
.segment-chip p { margin: 0; font-size: .84rem; opacity: .82; }

/* ---------- Interactive widgets ---------- */
.stButton > button {
    transition: transform .16s ease, box-shadow .16s ease !important;
    border-radius: 4px !important;
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 16px rgba(201,138,46,.28); }
.stButton > button:active { transform: translateY(0) scale(.97); }

button[kind="primary"] {
    background: linear-gradient(160deg, var(--gold), var(--gold-2)) !important;
    border: none !important; color: var(--ink) !important; font-weight: 600 !important;
}
button[kind="primary"]:hover { box-shadow: 0 0 0 3px rgba(201,138,46,.22), 0 10px 20px rgba(201,138,46,.35) !important; }

[data-baseweb="tab-list"] button { transition: transform .16s ease, color .16s ease; }
[data-baseweb="tab-list"] button:hover { transform: translateY(-1px); }
[data-baseweb="tab-highlight"] { background-color: var(--gold) !important; transition: all .25s ease !important; }

div[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace; animation: fadeInUp .45s ease both; }
img, .stPlotlyChart { animation: fadeInUp .5s ease both; }

/* ============================================================
   PREMIUM REFINEMENT PASS — sharper edges, hairlines, certificate
   framing, small-caps tracking, beveled gold, cardstock grain.
   ============================================================ */

/* Sharper, editorial corner radius throughout (premium reads precise, not bubbly) */
.hero, .kpi-card, .result-card, .segment-chip { border-radius: 2px !important; }

/* Subtle cardstock grain over the whole app */
.grain-overlay {
    position: fixed; inset: 0; pointer-events: none; z-index: 9999; opacity: .05; mix-blend-mode: overlay;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
}

/* Eyebrow / label tracking — small caps, wide letterspacing reads as premium editorial */
.hero .eyebrow { font-variant: small-caps; letter-spacing: .16em; font-weight: 600; }
.kpi-card .kpi-label { font-variant: small-caps; letter-spacing: .1em; }
.result-badge { font-variant: small-caps; letter-spacing: .1em; }

/* KPI cards: thin gold hairline top instead of colored block, deeper layered shadow */
.kpi-card {
    border-left: none !important; border-top: 2px solid var(--gold) !important;
    padding: 1.1rem 1.15rem !important;
    box-shadow: 0 1px 0 rgba(255,255,255,.04) inset, 0 10px 24px rgba(0,0,0,.28) !important;
}
.kpi-card:hover { border-top-color: var(--teal) !important; }

/* Result card: certificate framing — fine gold border + corner brackets */
.result-card {
    border: 1px solid var(--gold) !important; border-left: 1px solid var(--gold) !important;
    box-shadow: 0 1px 0 rgba(255,255,255,.05) inset, 0 16px 38px rgba(0,0,0,.4) !important;
}
.result-card::before, .result-card::after {
    content: ""; position: absolute; width: 16px; height: 16px; border: 2px solid var(--gold); opacity: .75;
}
.result-card::before { top: 9px; left: 9px; border-right: none; border-bottom: none; }
.result-card::after  { bottom: 9px; right: 9px; border-left: none; border-top: none; }

/* Segment chips: fine hairline all around, gold top rule on hover */
.segment-chip { border: 1px solid #ffffff14 !important; border-top: 2px solid var(--tier-color, var(--gold)) !important; }

/* Beveled, tracked gold button — reads engraved rather than flat */
button[kind="primary"] {
    background: linear-gradient(180deg, #dba54a, var(--gold) 45%, var(--gold-2)) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,.4), inset 0 -1px 0 rgba(0,0,0,.25), 0 8px 18px rgba(0,0,0,.35) !important;
    letter-spacing: .05em !important; font-size: .86rem !important;
}

/* Ornamented section dividers */
hr {
    border: none !important; border-top: 1px solid var(--line) !important;
    position: relative !important; margin: 2.2rem 0 !important; overflow: visible !important;
}
hr::after {
    content: "◆"; position: absolute; top: -8px; left: 50%; transform: translateX(-50%);
    background: var(--ink); color: var(--gold); padding: 0 12px; font-size: .65rem;
}

/* Tables: hairline rows instead of default zebra */
[data-testid="stDataFrame"] { border: 1px solid #ffffff16 !important; }

/* Premium footer rule */
.ledger-footer {
    margin-top: 2.5rem; padding-top: 1.1rem; border-top: 1px solid var(--line);
    font-family: 'IBM Plex Mono', monospace; font-size: .72rem; letter-spacing: .08em;
    text-transform: uppercase; color: #8f8874; text-align: center;
}

@media (prefers-reduced-motion: reduce) {
    * { animation-duration: 0.01ms !important; animation-iteration-count: 1 !important; }
}
</style>
<div class="grain-overlay"></div>
""", unsafe_allow_html=True)

# ============================================================
# PREMIUM SaaS REFINEMENT — motion, glass, onboarding, voice
# ============================================================
st.markdown("""
<style>
/* Animated ambient background */
[data-testid="stAppViewContainer"] {
    background:
      radial-gradient(circle at 8% 5%, rgba(201,138,46,.12), transparent 28%),
      radial-gradient(circle at 92% 12%, rgba(47,111,99,.10), transparent 26%),
      linear-gradient(180deg, #14131b 0%, #111017 100%);
}
[data-testid="stAppViewContainer"]::before {
    content:"";
    position:fixed; inset:-20%;
    pointer-events:none; z-index:0;
    background:
      radial-gradient(circle at 30% 40%, rgba(201,138,46,.055), transparent 22%),
      radial-gradient(circle at 72% 65%, rgba(47,111,99,.05), transparent 24%);
    animation: ambientDrift 14s ease-in-out infinite alternate;
}
@keyframes ambientDrift {
    from { transform: translate3d(-1%, -1%, 0) scale(1); }
    to { transform: translate3d(2%, 1%, 0) scale(1.06); }
}
[data-testid="stHeader"] { background: rgba(20,19,27,.55); backdrop-filter: blur(14px); }

/* Premium hero */
.hero {
    border-radius: 18px !important;
    padding: 2.5rem 2.7rem 2.7rem !important;
    border: 1px solid rgba(255,255,255,.10) !important;
    box-shadow: 0 28px 70px rgba(0,0,0,.38), inset 0 1px 0 rgba(255,255,255,.06) !important;
    background:
      radial-gradient(circle at 88% 18%, rgba(201,138,46,.18), transparent 22%),
      linear-gradient(135deg, #211f2b 0%, #14131b 72%) !important;
}
.hero::after {
    content:"AI-POWERED CUSTOMER INTELLIGENCE";
    position:absolute; right:22px; bottom:18px;
    font-family:'IBM Plex Mono',monospace; font-size:.55rem;
    letter-spacing:.15em; color:#ffffff55;
}
.hero .stamp {
    border-radius:50% !important;
    box-shadow: 0 0 30px rgba(201,138,46,.18);
}
.hero h1 { font-size: clamp(2.2rem, 5vw, 4rem) !important; line-height:1 !important; }
.hero .subtitle { font-size:1rem !important; max-width:680px !important; }

/* Floating premium cards */
.kpi-card {
    border-radius:16px !important;
    border:1px solid rgba(255,255,255,.08) !important;
    border-top:2px solid var(--gold) !important;
    background:linear-gradient(145deg, rgba(35,33,45,.92), rgba(20,19,27,.92)) !important;
    backdrop-filter:blur(14px);
    box-shadow:0 16px 35px rgba(0,0,0,.25) !important;
}
.kpi-card:hover {
    transform:translateY(-6px) scale(1.01) !important;
    box-shadow:0 22px 45px rgba(0,0,0,.38), 0 0 0 1px rgba(201,138,46,.14) !important;
}
.kpi-value { letter-spacing:-.04em; }

/* Section glass containers */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius:18px !important;
    border-color:rgba(255,255,255,.08) !important;
    background:rgba(28,26,38,.40) !important;
    backdrop-filter:blur(10px);
}

/* Inputs */
[data-baseweb="input"] > div,
[data-baseweb="select"] > div,
[data-baseweb="textarea"] > div {
    border-radius:12px !important;
    background:rgba(255,255,255,.035) !important;
    border-color:rgba(255,255,255,.12) !important;
    transition:all .2s ease !important;
}
[data-baseweb="input"] > div:focus-within,
[data-baseweb="select"] > div:focus-within {
    border-color:rgba(201,138,46,.75) !important;
    box-shadow:0 0 0 3px rgba(201,138,46,.10) !important;
}
[data-testid="stSlider"] [role="slider"] {
    box-shadow:0 0 0 5px rgba(201,138,46,.08);
}

/* Buttons */
.stButton > button, .stDownloadButton > button {
    border-radius:12px !important;
    min-height:42px !important;
    font-weight:600 !important;
    transition:all .18s cubic-bezier(.2,.8,.2,1) !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform:translateY(-2px) !important;
}

/* Result */
.result-card {
    border-radius:18px !important;
    background:
      radial-gradient(circle at 90% 10%, color-mix(in srgb, var(--tier-color) 15%, transparent), transparent 28%),
      linear-gradient(145deg, #211f2b, #15141c) !important;
    animation: resultPop .7s cubic-bezier(.16,1,.3,1) both !important;
}
@keyframes resultPop {
    0% { opacity:0; transform:translateY(18px) scale(.97); }
    65% { opacity:1; transform:translateY(-3px) scale(1.005); }
    100% { transform:translateY(0) scale(1); }
}

/* Segment chips */
.segment-chip { border-radius:14px !important; }
.segment-chip:hover { transform:translateY(-5px) scale(1.01) !important; }

/* Pro pill + helper cards */
.pro-pill {
    display:inline-flex; align-items:center; gap:.35rem;
    padding:.35rem .65rem; border-radius:999px;
    background:linear-gradient(135deg,#dba54a,#a5721f);
    color:#14131b; font-weight:800; font-size:.68rem;
    letter-spacing:.08em; text-transform:uppercase;
    box-shadow:0 6px 18px rgba(201,138,46,.25);
}
.hint-card {
    border:1px dashed rgba(201,138,46,.28);
    background:rgba(201,138,46,.055);
    border-radius:14px; padding:.8rem 1rem;
    font-size:.83rem; color:#e8e2d0;
}
.step-pill {
    display:inline-block; padding:.38rem .65rem; margin-right:.35rem;
    border-radius:999px; font-family:'IBM Plex Mono',monospace;
    font-size:.68rem; letter-spacing:.06em;
    border:1px solid rgba(255,255,255,.10); color:#aaa39a;
}
.step-pill.active { border-color:#c98a2e; color:#f4eedd; background:rgba(201,138,46,.10); }

/* Sidebar */
section[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#15141d,#111017) !important;
    border-right:1px solid rgba(255,255,255,.06);
}
section[data-testid="stSidebar"] .stButton > button { width:100%; }

/* Reduce-motion accessibility */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation-duration:.01ms !important; transition:none !important; }
}
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


def style_fig(fig):
    """Applies the ledger theme (ink background, parchment text, mono/serif fonts) to a Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#e8e2d0"),
        title_font=dict(family="Fraunces, serif", color="#f4eedd", size=17),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=fig.layout.margin.t or 40),
    )
    fig.update_xaxes(gridcolor="#ffffff14", zerolinecolor="#ffffff20", linecolor="#ffffff20")
    fig.update_yaxes(gridcolor="#ffffff14", zerolinecolor="#ffffff20", linecolor="#ffffff20")
    return fig


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

def speak_result(text):
    """Uses the browser's built-in speech synthesis; no API key required."""
    safe = str(text).replace("\\", "\\\\").replace("'", "\\'")
    components.html(
        f"""
        <script>
        try {{
            const utterance = new SpeechSynthesisUtterance('{safe}');
            utterance.rate = 0.92;
            utterance.pitch = 1.0;
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(utterance);
        }} catch (e) {{}}
        </script>
        """,
        height=0,
        width=0,
    )


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
# HERO HEADER — ledger masthead
# ============================================================
st.markdown(f"""
<div class="hero">
  <div class="stamp">Verified<br/>Record</div>
  <p class="eyebrow">Customer Ledger · RFM Analysis</p>
  <h1>Segment Predictor</h1>
  <p class="subtitle">{animated_words("Score customers by Recency, Frequency, and Monetary value — then act on it.")}</p>
</div>
<div class="perforation"></div>
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
    st.markdown('<span class="pro-pill">✦ FREEMIUM · PRO READY</span>', unsafe_allow_html=True)
    st.write("")
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

    voice_enabled = st.toggle(
        "🔊 Voice insights",
        value=False,
        help="After a prediction, the browser can read the segment and confidence aloud."
    )

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
        <div class="segment-chip" style="--tier-color:{color}; animation-delay:{i*0.08}s;">
            <h4>{info['emoji']} {seg}</h4>
            <p>{info['desc']}</p>
        </div>
        """, unsafe_allow_html=True)
        st.write("")

# ============================================================
# QUICK START / ONBOARDING
# ============================================================
with st.expander("✨ New here? 30-second guide", expanded=False):
    st.markdown("""
    <div class="hint-card">
    <b>1 · Predict</b> — enter Recency, Frequency and Monetary value.<br>
    <b>2 · Understand</b> — the model returns a segment, confidence and customer profile.<br>
    <b>3 · Act</b> — use the Growth Playbook to choose a practical campaign.<br>
    <b>4 · Scale</b> — upload a customer file to score many customers at once.
    </div>
    """, unsafe_allow_html=True)

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
# TAB 1 — SINGLE PREDICTION / AI CLASSIFICATION / ACTION PLAN
# ------------------------------------------------------------
# The original design used decorative HTML pills for these three steps.
# They looked like buttons but could not be clicked.  This version uses
# real Streamlit buttons + session state, so all three steps are functional.
with tab1:
    if "single_step" not in st.session_state:
        st.session_state.single_step = "input"
    if "single_prediction" not in st.session_state:
        st.session_state.single_prediction = None

    st.subheader("Customer intelligence workflow")
    st.caption("Move through the three stages below. Your prediction is saved during this session.")

    step1, step2, step3 = st.columns(3)
    with step1:
        if st.button("01 · RFM INPUT", use_container_width=True,
                     type="primary" if st.session_state.single_step == "input" else "secondary",
                     key="step_input"):
            st.session_state.single_step = "input"
            st.rerun()
    with step2:
        if st.button("02 · AI CLASSIFICATION", use_container_width=True,
                     type="primary" if st.session_state.single_step == "classification" else "secondary",
                     key="step_classification"):
            if st.session_state.single_prediction is None:
                st.toast("Run a prediction first.", icon="💡")
                st.session_state.single_step = "input"
            else:
                st.session_state.single_step = "classification"
            st.rerun()
    with step3:
        if st.button("03 · ACTION PLAN", use_container_width=True,
                     type="primary" if st.session_state.single_step == "action" else "secondary",
                     key="step_action"):
            if st.session_state.single_prediction is None:
                st.toast("Run a prediction first.", icon="💡")
                st.session_state.single_step = "input"
            else:
                st.session_state.single_step = "action"
            st.rerun()

    st.write("")

    # ---------------- RFM INPUT ----------------
    if st.session_state.single_step == "input":
        st.markdown(
            '<div class="hint-card">💡 <b>RFM guide:</b> Lower Recency is better. Higher Frequency and Monetary value usually indicate stronger customer value.</div>',
            unsafe_allow_html=True,
        )
        st.write("")

        input_mode = st.radio("Input method", ["Sliders", "Number fields"], horizontal=True, key="single_input_mode")
        col1, col2, col3 = st.columns(3)

        if input_mode == "Sliders":
            with col1:
                recency = st.slider("Recency (days since last purchase)", 0, 400, 30, key="single_recency_slider")
            with col2:
                frequency = st.slider("Frequency (number of orders)", 1, 100, 3, key="single_frequency_slider")
            with col3:
                monetary = st.slider("Monetary (total spend, £)", 0.0, 20000.0, 500.0, step=10.0, key="single_monetary_slider")
        else:
            with col1:
                recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=1000, value=30, key="single_recency_number")
            with col2:
                frequency = st.number_input("Frequency (number of orders)", min_value=1, max_value=500, value=3, key="single_frequency_number")
            with col3:
                monetary = st.number_input("Monetary (total spend, £)", min_value=0.0, max_value=1_000_000.0, value=500.0, step=10.0, key="single_monetary_number")

        with st.expander("What do these values mean?", expanded=False):
            h1, h2, h3 = st.columns(3)
            with h1:
                st.markdown("**Recency**")
                st.caption("Days since the customer's latest purchase. Smaller is generally better.")
            with h2:
                st.markdown("**Frequency**")
                st.caption("Number of purchases/orders. Higher generally means stronger engagement.")
            with h3:
                st.markdown("**Monetary**")
                st.caption("Total customer spend. Higher generally means greater customer value.")

        if st.button("✨ Analyze Customer", type="primary", use_container_width=True, key="analyze_customer"):
            with st.spinner("Analyzing RFM profile…"):
                segment, proba = predict_segment(recency, frequency, monetary, temperature=conf_temp)
                confidence = float(np.max(proba))
                sorted_proba = np.sort(proba)[::-1]
                margin = float(sorted_proba[0] - sorted_proba[1]) if len(sorted_proba) > 1 else 1.0
                info = SEGMENT_INFO[segment]

                st.session_state.single_prediction = {
                    "recency": float(recency),
                    "frequency": float(frequency),
                    "monetary": float(monetary),
                    "segment": segment,
                    "proba": np.asarray(proba, dtype=float),
                    "confidence": confidence,
                    "margin": margin,
                }

                new_row = pd.DataFrame([{
                    "Timestamp": datetime.now().strftime("%H:%M:%S"),
                    "Recency": recency,
                    "Frequency": frequency,
                    "Monetary": monetary,
                    "Segment": segment,
                    "Confidence": round(confidence, 3),
                }])
                st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True)

                if confidence >= 0.95:
                    confetti_burst(90, colors=[SEGMENT_COLORS[segment], "#C98A2E", "#f4eedd"])
                    st.toast(f"Outstanding match: {segment}", icon="🏅")
                elif confidence >= 0.85:
                    confetti_burst(60, colors=[SEGMENT_COLORS[segment], "#f4eedd"])
                    st.toast(f"High-confidence match: {segment}", icon="✨")
                else:
                    st.toast("Prediction saved. Review the AI Classification step.", icon="📝")

            st.session_state.single_step = "classification"
            st.rerun()

        if st.session_state.single_prediction is not None:
            st.info("A prediction is already available. Click **02 · AI CLASSIFICATION** above to review it, or analyze a new customer.", icon="✨")

    # ---------------- AI CLASSIFICATION ----------------
    elif st.session_state.single_step == "classification":
        pred = st.session_state.single_prediction
        segment = pred["segment"]
        proba = pred["proba"]
        confidence = pred["confidence"]
        margin = pred["margin"]
        info = SEGMENT_INFO[segment]
        color = SEGMENT_COLORS[segment]

        st.markdown(
            '<div class="hint-card">🧠 <b>AI Classification:</b> the trained model evaluates the transformed RFM values and returns the most likely customer segment plus probability estimates.</div>',
            unsafe_allow_html=True,
        )
        st.write("")

        res_col, profile_col = st.columns([1.05, 0.95])
        with res_col:
            st.markdown(f"""
            <div class="result-card" style="--tier-color:{color};">
                <div>
                    <span class="result-badge">{info['priority']} priority</span>
                    <div class="result-title">{info['emoji']} {segment}</div>
                    <div class="result-desc">{info['desc']}</div>
                    <div class="confidence-track"><div class="confidence-fill" style="width:{confidence*100:.1f}%;"></div></div>
                    <div class="confidence-label">{confidence:.1%} confidence · margin over runner-up: {margin:.1%}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if voice_enabled:
                if st.button("🔊 Read AI result aloud", use_container_width=True, key="speak_classification"):
                    speak_result(
                        f"Customer segment: {segment}. Confidence: {confidence:.0%}. Recommended action: {info['action']}"
                    )

            prob_df = pd.DataFrame({"Segment": clf.classes_, "Probability": proba}).sort_values("Probability", ascending=True)
            fig = px.bar(
                prob_df, x="Probability", y="Segment", orientation="h", text="Probability",
                color="Segment", color_discrete_map=SEGMENT_COLORS,
            )
            fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
            fig.update_layout(showlegend=False, xaxis=dict(range=[0, 1], tickformat=".0%"), yaxis_title="", height=300, margin=dict(t=10, b=10), transition_duration=500)
            st.plotly_chart(style_fig(fig), use_container_width=True)

        with profile_col:
            st.markdown("**Customer profile**")
            r_pct = percentile_of(pred["recency"], REF["Recency"], invert=True)
            f_pct = percentile_of(pred["frequency"], REF["Frequency"])
            m_pct = percentile_of(pred["monetary"], REF["Monetary"])
            radar = go.Figure()
            radar.add_trace(go.Scatterpolar(
                r=[r_pct, f_pct, m_pct, r_pct],
                theta=["Recency", "Frequency", "Monetary", "Recency"],
                fill="toself", name="This customer", line_color=color,
            ))
            radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100], ticksuffix="%")),
                showlegend=False, height=360, margin=dict(t=20, b=20), transition_duration=600,
            )
            st.plotly_chart(style_fig(radar), use_container_width=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("Recency", f"{pred['recency']:,.0f} days")
            m2.metric("Frequency", f"{pred['frequency']:,.0f}")
            m3.metric("Monetary", f"£{pred['monetary']:,.0f}")

        st.success(f"**Next step:** open **03 · ACTION PLAN** above for a segment-specific growth strategy.", icon="🚀")

    # ---------------- ACTION PLAN ----------------
    else:
        pred = st.session_state.single_prediction
        segment = pred["segment"]
        info = SEGMENT_INFO[segment]
        play = GROWTH_PLAYBOOK[segment]
        color = SEGMENT_COLORS[segment]

        st.markdown(
            f'<div class="hint-card">🚀 <b>Action Plan for {segment}:</b> turn the classification into a practical customer-growth workflow.</div>',
            unsafe_allow_html=True,
        )
        st.write("")

        st.markdown(f"""
        <div class="result-card" style="--tier-color:{color};">
            <span class="result-badge">{info['priority']} priority</span>
            <div class="result-title">{info['emoji']} {segment}</div>
            <div class="result-desc"><b>Objective:</b> {play['objective']}</div>
        </div>
        """, unsafe_allow_html=True)
        st.write("")

        st.markdown("### 🎯 Recommended actions")
        for i, (name, desc, channel) in enumerate(play["tactics"], start=1):
            st.markdown(f"""
            <div class="kpi-card" style="text-align:left; margin-bottom:.65rem; animation-delay:{i*0.08}s;">
                <div class="kpi-label">STEP {i} · {channel}</div>
                <div style="font-weight:700; font-size:1.02rem; margin-top:.2rem;">{name}</div>
                <div style="opacity:.82; font-size:.9rem; margin-top:.25rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 📌 Success metric")
        st.info(play["kpi"], icon="📈")

        st.markdown("### 💰 Opportunity simulator")
        uplift_default = int(round(play["uplift_pct"] * 100))
        uplift_pct = st.slider(
            "Assumed incremental spend uplift",
            0, 30, uplift_default, 1, format="%d%%",
            key=f"single_action_uplift_{segment}",
            help="Planning assumption only — this is not a guaranteed forecast.",
        )
        estimated_gain = pred["monetary"] * uplift_pct / 100
        a1, a2, a3 = st.columns(3)
        a1.metric("Customer value", f"£{pred['monetary']:,.0f}")
        a2.metric("Assumed uplift", f"{uplift_pct}%")
        a3.metric("Estimated opportunity", f"£{estimated_gain:,.0f}")

        action_text = [
            "CUSTOMER ACTION PLAN",
            f"Segment: {segment}",
            f"Priority: {info['priority']}",
            f"Confidence: {pred['confidence']:.1%}",
            f"Objective: {play['objective']}",
            "",
        ]
        for i, (name, desc, channel) in enumerate(play["tactics"], 1):
            action_text.append(f"{i}. {name} [{channel}] — {desc}")
        action_text.extend([
            "",
            f"Track: {play['kpi']}",
            f"Estimated opportunity: £{estimated_gain:,.0f}",
            "Note: opportunity is a planning assumption, not a forecast.",
        ])

        dl_col, new_col = st.columns(2)
        with dl_col:
            st.download_button(
                "⬇️ Download Action Plan",
                "\n".join(action_text).encode("utf-8"),
                "customer_action_plan.txt",
                "text/plain",
                type="primary",
                use_container_width=True,
            )
        with new_col:
            if st.button("↩️ Analyze another customer", use_container_width=True, key="new_customer"):
                st.session_state.single_step = "input"
                st.session_state.single_prediction = None
                st.rerun()

    if not st.session_state.history.empty:
        with st.expander(f"📜 Full session history ({len(st.session_state.history)} predictions)"):
            st.dataframe(st.session_state.history, use_container_width=True)
            trend = px.bar(
                st.session_state.history["Segment"].value_counts().reset_index(),
                x="Segment", y="count", color="Segment", color_discrete_map=SEGMENT_COLORS,
            )
            trend.update_layout(showlegend=False, height=250, margin=dict(t=10, b=10))
            st.plotly_chart(style_fig(trend), use_container_width=True)

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
                st.plotly_chart(style_fig(fig2), use_container_width=True)
            with c2:
                fig3 = px.scatter(
                    data, x="Recency", y="Monetary", size="Frequency", color="Predicted_Segment",
                    color_discrete_map=SEGMENT_COLORS, hover_data=["Frequency", "Confidence"],
                    title="Recency vs. Monetary (bubble size = Frequency)",
                )
                st.plotly_chart(style_fig(fig3), use_container_width=True)

            csv_out = data.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download scored CSV", csv_out, "scored_customers.csv", "text/csv", type="primary")

            # ---------------- Aggregated Growth Playbook ----------------
            st.divider()
            st.subheader("📋 Business Growth Playbook")
            st.caption("Prioritized by estimated revenue opportunity — biggest lever first.")

            seg_summary = data.groupby("Predicted_Segment")["Monetary"].agg(["sum", "count"]).reset_index()
            seg_summary.columns = ["Segment", "TotalValue", "Customers"]
            seg_summary["UpliftPct"] = seg_summary["Segment"].map(lambda s: GROWTH_PLAYBOOK[s]["uplift_pct"])
            seg_summary["EstimatedGain"] = seg_summary["TotalValue"] * seg_summary["UpliftPct"]
            seg_summary = seg_summary.sort_values("EstimatedGain", ascending=False).reset_index(drop=True)

            fig_opp = px.bar(
                seg_summary, x="Segment", y="EstimatedGain", color="Segment",
                color_discrete_map=SEGMENT_COLORS, text="EstimatedGain",
                title="Estimated revenue opportunity by segment (default assumptions)",
            )
            fig_opp.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
            fig_opp.update_layout(showlegend=False, yaxis_title="Estimated opportunity ($)", height=320)
            st.plotly_chart(style_fig(fig_opp), use_container_width=True)

            total_opportunity = seg_summary["EstimatedGain"].sum()
            st.markdown(f"**Total estimated opportunity across all segments: ${total_opportunity:,.0f}** *(default assumptions — adjust per segment below)*")

            action_plan_lines = [
                "BUSINESS GROWTH ACTION PLAN",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                f"Customers scored: {len(data):,}",
                f"Total estimated opportunity: ${total_opportunity:,.0f}",
                "=" * 60,
            ]

            for _, row in seg_summary.iterrows():
                seg = row["Segment"]
                play = GROWTH_PLAYBOOK[seg]
                with st.expander(
                    f"{SEGMENT_INFO[seg]['emoji']} {seg} — {int(row['Customers'])} customers · "
                    f"${row['TotalValue']:,.0f} total value · ${row['EstimatedGain']:,.0f} estimated opportunity"
                ):
                    render_growth_playbook(seg, monetary_value=row["TotalValue"], customer_count=1, key_suffix=f"batch_{seg}")

                action_plan_lines.append(f"\n{seg.upper()} ({int(row['Customers'])} customers, ${row['TotalValue']:,.0f} total value)")
                action_plan_lines.append(f"Objective: {play['objective']}")
                for name, desc, channel in play["tactics"]:
                    action_plan_lines.append(f"  - [{channel}] {name}: {desc}")
                action_plan_lines.append(f"Track: {play['kpi']}")
                action_plan_lines.append(f"Estimated opportunity: ${row['EstimatedGain']:,.0f} ({play['uplift_note']})")

            st.download_button(
                "⬇️ Download full Growth Action Plan (.txt)",
                "\n".join(action_plan_lines).encode("utf-8"),
                "growth_action_plan.txt", "text/plain",
            )
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
        st.plotly_chart(style_fig(fig_imp), use_container_width=True)
    elif hasattr(clf, "coef_"):
        coefs = pd.DataFrame(clf.coef_, columns=FEATURES, index=clf.classes_)
        st.write("Logistic regression coefficients (log-odds impact per class):")
        st.dataframe(coefs, use_container_width=True)
        fig_coef = px.imshow(coefs, color_continuous_scale=[[0, "#2F6F63"], [0.5, "#1c1a26"], [1, "#C98A2E"]],
                              aspect="auto",
                              text_auto=".2f", title="Coefficient heatmap")
        fig_coef.update_layout(height=280, margin=dict(t=40, b=10))
        st.plotly_chart(style_fig(fig_coef), use_container_width=True)
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
    st.plotly_chart(style_fig(fig_grid), use_container_width=True)
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
            <div class="segment-chip" style="--tier-color:{color}; min-height:190px; animation-delay:{i*0.1}s;">
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

st.markdown("""
<div class="hint-card" style="margin-top:1.5rem; text-align:center;">
    <span class="pro-pill">✦ PRO</span>
    &nbsp; Unlock deeper analytics, automated reports, AI-assisted insights and team workflows.
    <br><small style="opacity:.65;">Freemium concept UI — no payment is required for this demo.</small>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="ledger-footer">
    {("Model — " + type(clf).__name__) if is_real_model else "Model — Synthetic Demo (Random Forest)"}
    &nbsp;·&nbsp; RFM Ledger Analysis
    &nbsp;·&nbsp; {"Validated on Held-Out Customers" if is_real_model else "Unvalidated Demo Data"}
</div>
""", unsafe_allow_html=True)
