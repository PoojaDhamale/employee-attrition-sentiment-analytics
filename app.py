import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import html
import os
import json
import plotly.express as px

# ----------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM AESTHETICS (CSS)
# ----------------------------------------------------
st.set_page_config(
    page_title="Employee Attrition & Sentiment Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"], .stApp, .stMarkdown, p, span, label, div {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    .stApp {
        background: linear-gradient(160deg, #06060e 0%, #0a0e1f 30%, #0d1225 60%, #080c18 100%);
        color: #e2e8f0;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .hero-header {
        text-align: center;
        padding: 3rem 2rem 2.5rem;
        background: linear-gradient(135deg, rgba(99,102,241,0.08) 0%, rgba(168,85,247,0.05) 50%, rgba(236,72,153,0.08) 100%);
        border-radius: 20px;
        margin-bottom: 2rem;
        border: 1px solid rgba(99,102,241,0.15);
        backdrop-filter: blur(20px);
        position: relative;
        overflow: hidden;
    }

    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 30% 20%, rgba(99,102,241,0.06) 0%, transparent 50%),
                    radial-gradient(circle at 70% 80%, rgba(236,72,153,0.04) 0%, transparent 50%);
        animation: headerGlow 15s ease-in-out infinite;
    }

    @keyframes headerGlow {
        0%, 100% { transform: translate(0, 0) rotate(0deg); }
        33% { transform: translate(3%, -2%) rotate(1deg); }
        66% { transform: translate(-2%, 3%) rotate(-1deg); }
    }

    .hero-title {
        font-size: 2.7rem;
        font-weight: 900;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #818cf8 0%, #a78bfa 25%, #c084fc 50%, #e879f9 75%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.6rem;
        position: relative;
        z-index: 1;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        font-weight: 400;
        letter-spacing: 0.01em;
        position: relative;
        z-index: 1;
        line-height: 1.6;
    }

    .hero-badge {
        display: inline-block;
        background: rgba(99,102,241,0.12);
        border: 1px solid rgba(99,102,241,0.25);
        padding: 6px 16px;
        border-radius: 100px;
        font-size: 0.75rem;
        font-weight: 600;
        color: #a5b4fc;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 1rem;
        position: relative;
        z-index: 1;
    }

    div[data-testid="stVerticalBlockBorderWrapper"], div[data-testid="stForm"] {
        background: rgba(15, 20, 40, 0.65) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(99,102,241,0.1) !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.03) !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:hover, div[data-testid="stForm"]:hover {
        border-color: rgba(99,102,241,0.2) !important;
        box-shadow: 0 8px 32px rgba(99,102,241,0.08), inset 0 1px 0 rgba(255,255,255,0.04) !important;
    }

    .card-label {
        font-size: 0.75rem;
        font-weight: 700;
        color: #818cf8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .card-label::after {
        content: '';
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, rgba(99,102,241,0.25) 0%, transparent 100%);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 20, 40, 0.6);
        backdrop-filter: blur(12px);
        padding: 8px 10px;
        border-radius: 14px;
        border: 1px solid rgba(99,102,241,0.1);
        margin-bottom: 1.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 10px;
        color: #64748b;
        font-weight: 600;
        font-size: 0.95rem;
        border: none;
        padding: 0 20px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #c4b5fd;
        background-color: rgba(99, 102, 241, 0.06);
    }

    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%) !important;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.35), 0 0 0 1px rgba(99,102,241,0.2);
    }

    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stTabs [data-baseweb="tab-border"] { display: none; }

    div[class*="stNumberInput"] input {
        background-color: rgba(15, 20, 45, 0.8) !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(99,102,241,0.15) !important;
        border-radius: 10px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.9rem !important;
        padding: 10px 14px !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
    }

    div[class*="stNumberInput"] input:focus {
        border-color: rgba(99,102,241,0.5) !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
    }

    div[class*="stTextInput"] input {
        background-color: rgba(15, 20, 45, 0.8) !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(99,102,241,0.15) !important;
        border-radius: 10px !important;
        font-size: 0.9rem !important;
        padding: 10px 14px !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
    }

    div[class*="stTextInput"] input:focus {
        border-color: rgba(99,102,241,0.5) !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
    }

    div[data-baseweb="select"] > div {
        background-color: rgba(15, 20, 45, 0.8) !important;
        border: 1px solid rgba(99,102,241,0.15) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
        transition: border-color 0.3s ease !important;
    }

    div[data-baseweb="select"] > div:hover {
        border-color: rgba(99,102,241,0.3) !important;
    }

    div[data-baseweb="popover"] > div {
        background-color: #0f1429 !important;
        border: 1px solid rgba(99,102,241,0.2) !important;
        border-radius: 12px !important;
    }

    ul[role="listbox"] li { color: #e2e8f0 !important; }
    ul[role="listbox"] li:hover { background-color: rgba(99,102,241,0.15) !important; }

    div[class*="stTextArea"] textarea {
        background-color: rgba(15, 20, 45, 0.8) !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(99,102,241,0.15) !important;
        border-radius: 12px !important;
        font-size: 0.95rem !important;
        line-height: 1.6 !important;
        padding: 14px !important;
        transition: border-color 0.3s ease !important;
    }

    div[class*="stTextArea"] textarea:focus {
        border-color: rgba(99,102,241,0.5) !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
    }

    div[class*="stSlider"] { padding-top: 8px; padding-bottom: 8px; }

    div[class*="stSlider"] [data-baseweb="slider"] div[role="slider"] {
        background: linear-gradient(135deg, #6366f1, #a855f7) !important;
        border: 2px solid #818cf8 !important;
    }

    .stNumberInput label, .stSelectbox label, .stTextArea label, .stTextInput label, .stSlider label {
        color: #94a3b8 !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
    }

    .stFormSubmitButton > button,
    div.stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 32px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        letter-spacing: 0.02em !important;
        box-shadow: 0 4px 20px rgba(99,102,241,0.3) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    .stFormSubmitButton > button:hover,
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(99,102,241,0.45) !important;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #9333ea 100%) !important;
    }

    .stFormSubmitButton > button:active,
    div.stButton > button:active { transform: translateY(0px) !important; }

    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700 !important;
        font-size: 2rem !important;
        background: linear-gradient(135deg, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
    }

    .kpi-card {
        background: rgba(15, 20, 45, 0.7);
        border-radius: 14px;
        padding: 18px 20px;
        text-align: left;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover { transform: translateY(-2px); }
    .kpi-label { font-size: 0.8rem; color: #94a3b8; font-weight: 500; margin-bottom: 8px; }
    .kpi-value { font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 1.9rem; }

    .kpi-neutral { border: 1px solid rgba(99,102,241,0.25); box-shadow: 0 0 16px rgba(99,102,241,0.06); }
    .kpi-neutral .kpi-value { color: #a5b4fc; }

    .kpi-positive { border: 1px solid rgba(34,197,94,0.25); box-shadow: 0 0 16px rgba(34,197,94,0.06); }
    .kpi-positive .kpi-value { color: #86efac; }

    .kpi-negative { border: 1px solid rgba(239,68,68,0.25); box-shadow: 0 0 16px rgba(239,68,68,0.06); }
    .kpi-negative .kpi-value { color: #fca5a5; }

    .result-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 10px 20px;
        border-radius: 100px;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 0.02em;
        animation: fadeInUp 0.5s ease-out;
    }

    .badge-high-risk {
        background: rgba(239,68,68,0.12);
        color: #fca5a5;
        border: 1px solid rgba(239,68,68,0.25);
        box-shadow: 0 0 20px rgba(239,68,68,0.1);
    }

    .badge-low-risk {
        background: rgba(34,197,94,0.12);
        color: #86efac;
        border: 1px solid rgba(34,197,94,0.25);
        box-shadow: 0 0 20px rgba(34,197,94,0.1);
    }

    .badge-medium-risk {
        background: rgba(245,158,11,0.12);
        color: #fcd34d;
        border: 1px solid rgba(245,158,11,0.25);
        box-shadow: 0 0 20px rgba(245,158,11,0.1);
    }

    .badge-positive {
        background: rgba(34,197,94,0.12);
        color: #86efac;
        border: 1px solid rgba(34,197,94,0.25);
        box-shadow: 0 0 20px rgba(34,197,94,0.1);
    }

    .badge-negative {
        background: rgba(239,68,68,0.12);
        color: #fca5a5;
        border: 1px solid rgba(239,68,68,0.25);
        box-shadow: 0 0 20px rgba(239,68,68,0.1);
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .result-title {
        font-size: 0.75rem;
        font-weight: 700;
        color: #818cf8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .result-title::after {
        content: '';
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, rgba(99,102,241,0.25) 0%, transparent 100%);
    }

    .stDataFrame { border-radius: 12px !important; overflow: hidden; }

    div[data-testid="stAlert"] {
        background: rgba(15, 20, 40, 0.6) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(99,102,241,0.15) !important;
    }

    .highlight-box {
        background: rgba(15, 20, 45, 0.8);
        border: 1px solid rgba(99,102,241,0.15);
        padding: 18px;
        border-radius: 12px;
        font-size: 0.95rem;
        line-height: 1.7;
        color: #e2e8f0;
        max-height: 280px;
        overflow-y: auto;
    }

    .highlight-box::-webkit-scrollbar { width: 6px; }
    .highlight-box::-webkit-scrollbar-track { background: rgba(15, 20, 40, 0.3); border-radius: 3px; }
    .highlight-box::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.3); border-radius: 3px; }

    .rec-list { list-style: none; padding: 0; margin: 0; }
    .rec-list li {
        padding: 10px 0;
        border-bottom: 1px solid rgba(99,102,241,0.08);
        color: #cbd5e1;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    .rec-list li:last-child { border-bottom: none; }
    .rec-list li strong { color: #a5b4fc; }

    .section-desc {
        color: #64748b;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 2. HELPER FUNCTIONS & MODEL LOADING
# ----------------------------------------------------
@st.cache_resource
def load_assets():
    ibm_model = joblib.load(os.path.join(BASE_DIR, 'attrition_model.pkl'))
    scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
    encoder = joblib.load(os.path.join(BASE_DIR, 'encoder.pkl'))
    feature_names = joblib.load(os.path.join(BASE_DIR, 'feature_names.pkl'))
    threshold = joblib.load(os.path.join(BASE_DIR, 'threshold.pkl'))
    nlp_pipeline = joblib.load(os.path.join(BASE_DIR, 'glassdoor_sentiment_pipeline.pkl'))
    firm_leaderboard = joblib.load(os.path.join(BASE_DIR, 'firm_sentiment_leaderboard.pkl'))
    return ibm_model, scaler, encoder, feature_names, threshold, nlp_pipeline, firm_leaderboard

try:
    ibm_model, scaler, encoder, feature_names, threshold, nlp_pipeline, firm_leaderboard = load_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}. Please ensure all pickle files are present in the workspace.")
    st.stop()

@st.cache_resource
def load_nlp_config():
    with open(os.path.join(BASE_DIR, 'glassdoor_nlp_config.json')) as f:
        return json.load(f)

config = load_nlp_config()

@st.cache_data
def load_ibm_raw():
    return pd.read_csv(os.path.join(BASE_DIR, "WA_Fn-UseC_-HR-Employee-Attrition.csv"))

ibm_raw = load_ibm_raw()

tfidf = nlp_pipeline.named_steps['tfidf']
clf = nlp_pipeline.named_steps['clf']
vocabulary = tfidf.get_feature_names_out()
coefficients = clf.coef_[0]
feature_coef_map = dict(zip(vocabulary, coefficients))

def get_html_highlighted_text(text, feature_coef_map, threshold=0.15):
    escaped_text = html.escape(text)
    text_lower = escaped_text.lower()

    matches = []
    for feature, coef in feature_coef_map.items():
        if abs(coef) >= threshold:
            pattern = rf"\b{re.escape(feature)}\b"
            for m in re.finditer(pattern, text_lower):
                matches.append((m.start(), m.end(), feature, coef))

    if not matches:
        return escaped_text

    matches = sorted(matches, key=lambda x: (x[0], -(x[1] - x[0])))
    clean_matches = []
    last_end = 0
    for start, end, feature, coef in matches:
        if start >= last_end:
            clean_matches.append((start, end, feature, coef))
            last_end = end

    parts = []
    last_idx = 0
    for start, end, feature, coef in clean_matches:
        parts.append(escaped_text[last_idx:start])
        word = escaped_text[start:end]
        if coef > 0:
            parts.append(f'<span style="background: rgba(34,197,94,0.2); color: #86efac; padding: 2px 6px; border-radius: 6px; font-weight: 600; border: 1px solid rgba(34,197,94,0.3);">{word}</span>')
        else:
            parts.append(f'<span style="background: rgba(239,68,68,0.2); color: #fca5a5; padding: 2px 6px; border-radius: 6px; font-weight: 600; border: 1px solid rgba(239,68,68,0.3);">{word}</span>')
        last_idx = end
    parts.append(escaped_text[last_idx:])

    return "".join(parts)

# ----------------------------------------------------
# 3. HEADER
# ----------------------------------------------------
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">📊 AI-Powered HR Analytics</div>
    <div class="hero-title">Employee Attrition &amp; Sentiment Analytics</div>
    <div class="hero-subtitle">Machine Learning Predictions for Structured Employee Attrition &amp; Unstructured Glassdoor Sentiment Analysis</div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 4. TABS DEFINITION
# ----------------------------------------------------
tab1, tab2 = st.tabs(["📊  Employee Attrition Predictor", "💬  Glassdoor Sentiment Analyzer"])

# ----------------------------------------------------
# TAB 1: EMPLOYEE ATTRITION PREDICTOR
# ----------------------------------------------------
with tab1:

    st.markdown('<div class="card-label">📊 Target Distribution</div>', unsafe_allow_html=True)
    st.markdown("""
    <p class="section-desc">
    Based on 1,470 employee records from the IBM HR dataset. The dataset is imbalanced —
    most employees stayed — so the model is evaluated using Precision, Recall, and
    AUC-ROC rather than relying on Accuracy alone.
    </p>
    """, unsafe_allow_html=True)

    attr_pct = ibm_raw['Attrition'].value_counts(normalize=True) * 100

    with st.container(border=True):
        e1, e2, e3 = st.columns(3)
        with e1:
            st.markdown(f"""
            <div class="kpi-card kpi-neutral">
                <div class="kpi-label">Total Employees</div>
                <div class="kpi-value">{len(ibm_raw):,}</div>
            </div>
            """, unsafe_allow_html=True)
        with e2:
            st.markdown(f"""
            <div class="kpi-card kpi-positive">
                <div class="kpi-label">Stayed</div>
                <div class="kpi-value">{attr_pct['No']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with e3:
            st.markdown(f"""
            <div class="kpi-card kpi-negative">
                <div class="kpi-label">Left</div>
                <div class="kpi-value">{attr_pct['Yes']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="card-label">🔥 Top Attrition Drivers</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-desc">Percentage of employees who left, broken down by category. Normalized by group size so small groups aren\'t misleading.</p>', unsafe_allow_html=True)

    with st.container(border=True):
        driver_col = st.selectbox(
            "Choose a category to break down:",
            ["OverTime", "Department", "JobRole", "BusinessTravel", "MaritalStatus", "Gender", "EducationField"]
        )

        driver_pct = (
            pd.crosstab(ibm_raw[driver_col], ibm_raw['Attrition'], normalize='index') * 100
        )['Yes'].sort_values().reset_index()
        driver_pct.columns = [driver_col, 'attrition_pct']

        fig_drivers = px.bar(
            driver_pct, x='attrition_pct', y=driver_col, orientation='h',
            labels={'attrition_pct': 'Attrition Rate (%)', driver_col: ''},
            color='attrition_pct', color_continuous_scale=['#34d399', '#f87171'],
            height=420
        )
        fig_drivers.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#cbd5e1", coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_drivers, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="card-label">📐 Key Numeric Differences</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-desc">How key numeric features differ between employees who stayed and those who left.</p>', unsafe_allow_html=True)

    with st.container(border=True):
        numeric_col = st.selectbox(
            "Choose a feature to compare:",
            ["MonthlyIncome", "Age", "YearsAtCompany", "TotalWorkingYears", "DistanceFromHome"]
        )

        fig_box = px.box(
            ibm_raw, x="Attrition", y=numeric_col, color="Attrition",
            color_discrete_map={"Yes": "#f87171", "No": "#34d399"},
            height=420
        )
        fig_box.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#cbd5e1", showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="card-label">🧮 Correlation Between Numeric Features</div>', unsafe_allow_html=True)
    st.markdown("""
    <p class="section-desc">
    Strongly correlated features (e.g. Age and TotalWorkingYears) can make individual
    coefficients harder to interpret in Logistic Regression — this is checked formally
    with VIF in the model build, shown here for reference.
    </p>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        num_cols_for_corr = ibm_raw.select_dtypes(include=['int64', 'float64']).columns.tolist()
        for drop_col in ['EmployeeCount', 'StandardHours', 'EmployeeNumber']:
            if drop_col in num_cols_for_corr:
                num_cols_for_corr.remove(drop_col)

        corr_matrix = ibm_raw[num_cols_for_corr].corr()

        fig_heat = px.imshow(
            corr_matrix, color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
            aspect="auto", height=600
        )
        fig_heat.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#cbd5e1",
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown('<div class="card-label">⚡ Predict Attrition Risk</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-desc">Enter the employee\'s operational metrics below to predict their risk of attrition. All numeric fields accept typed input.</p>', unsafe_allow_html=True)

    with st.form("attrition_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            with st.container(border=True):
                st.markdown('<div class="card-label">👤 Demographics & Base</div>', unsafe_allow_html=True)
                age = st.number_input("Age", min_value=18, max_value=60, value=35, step=1)
                gender = st.selectbox("Gender", ["Female", "Male"])
                marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
                education = st.selectbox("Education Level", [1, 2, 3, 4, 5],
                                         format_func=lambda x: {1: "1 — Below College", 2: "2 — College", 3: "3 — Bachelor", 4: "4 — Master", 5: "5 — Doctor"}[x])
                education_field = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other", "Human Resources"])

            with st.container(border=True):
                st.markdown('<div class="card-label">📋 Policy & Experience</div>', unsafe_allow_html=True)
                overtime = st.selectbox("Overtime Worked?", ["Yes", "No"])
                business_travel = st.selectbox("Business Travel Frequency", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
                distance_from_home = st.number_input("Distance From Home (miles)", min_value=1, max_value=30, value=8, step=1)
                num_companies_worked = st.number_input("Number of Companies Worked", min_value=0, max_value=10, value=2, step=1)

        with col2:
            with st.container(border=True):
                st.markdown('<div class="card-label">💼 Job & Career Metrics</div>', unsafe_allow_html=True)
                department = st.selectbox("Department", ["Research & Development", "Sales", "Human Resources"])
                job_role = st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"])
                job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
                training_times = st.number_input("Training Times Last Year", min_value=0, max_value=6, value=2, step=1)
                total_working_years = st.number_input("Total Working Years", min_value=0, max_value=40, value=10, step=1)

            with st.container(border=True):
                st.markdown('<div class="card-label">💰 Financial Metrics</div>', unsafe_allow_html=True)
                monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=25000, value=6500, step=500)
                salary_hike = st.number_input("Percent Salary Hike (%)", min_value=11, max_value=25, value=14, step=1)
                stock_option = st.selectbox("Stock Option Level", [0, 1, 2, 3])

        with col3:
            with st.container(border=True):
                st.markdown('<div class="card-label">📊 Satisfaction Scores (1–4)</div>', unsafe_allow_html=True)
                job_satisfaction = st.number_input("Job Satisfaction", min_value=1, max_value=4, value=3, step=1)
                env_satisfaction = st.number_input("Environment Satisfaction", min_value=1, max_value=4, value=3, step=1)
                rel_satisfaction = st.number_input("Relationship Satisfaction", min_value=1, max_value=4, value=3, step=1)
                wlb = st.number_input("Work Life Balance", min_value=1, max_value=4, value=3, step=1)
                job_involvement = st.number_input("Job Involvement", min_value=1, max_value=4, value=3, step=1)

            with st.container(border=True):
                st.markdown('<div class="card-label">🏢 Tenure at Company</div>', unsafe_allow_html=True)
                years_at_company = st.number_input("Years At Company", min_value=0, max_value=40, value=5, step=1)
                years_in_role = st.number_input("Years In Current Role", min_value=0, max_value=40, value=3, step=1)
                years_since_promotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=15, value=1, step=1)
                years_with_manager = st.number_input("Years With Current Manager", min_value=0, max_value=40, value=3, step=1)

        submit_btn = st.form_submit_button("⚡  Predict Attrition Risk", use_container_width=True)

    if submit_btn:
        total_satisfaction = np.mean([
            job_satisfaction,
            env_satisfaction,
            rel_satisfaction,
            wlb,
            job_involvement
        ])

        input_data = {
            'Age': age,
            'DistanceFromHome': distance_from_home,
            'Education': education,
            'Gender': 1 if gender == "Male" else 0,
            'JobLevel': job_level,
            'MonthlyIncome': monthly_income,
            'NumCompaniesWorked': num_companies_worked,
            'OverTime': 1 if overtime == "Yes" else 0,
            'PercentSalaryHike': salary_hike,
            'PerformanceRating': 4 if salary_hike >= 20 else 3,
            'StockOptionLevel': stock_option,
            'TotalWorkingYears': total_working_years,
            'TrainingTimesLastYear': training_times,
            'YearsAtCompany': years_at_company,
            'YearsInCurrentRole': years_in_role,
            'YearsSinceLastPromotion': years_since_promotion,
            'YearsWithCurrManager': years_with_manager,
            'Total_Satisfaction': total_satisfaction,
        }

        # Categorical columns are encoded using the SAME fitted OneHotEncoder
        # saved during training (encoder.pkl), instead of a hand-written
        # mapping -- this keeps the app and the notebook in sync even if
        # the encoder is ever refit on new categories.
        cat_input = pd.DataFrame([{
            'BusinessTravel': business_travel,
            'Department': department,
            'EducationField': education_field,
            'JobRole': job_role,
            'MaritalStatus': marital_status
        }])

        cat_cols_for_encoder = ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus']
        cat_encoded = encoder.transform(cat_input[cat_cols_for_encoder])
        cat_encoded_df = pd.DataFrame(
            cat_encoded,
            columns=encoder.get_feature_names_out(cat_cols_for_encoder)
        )

        df_input = pd.concat([pd.DataFrame([input_data]), cat_encoded_df], axis=1)

        for col in feature_names:
            if col not in df_input.columns:
                df_input[col] = 0

        df_input = df_input[feature_names]

        X_scaled = scaler.transform(df_input)

        prob = ibm_model.predict_proba(X_scaled)[0, 1]
        is_attrition = prob >= threshold

        # -----------------------------
        # Risk Level
        # -----------------------------
        if prob >= 0.75:
            risk_level = "High"
            badge_class = "badge-high-risk"
            badge_icon = "🚨"
        elif prob >= threshold:
            risk_level = "Moderate"
            badge_class = "badge-medium-risk"
            badge_icon = "⚠️"
        else:
            risk_level = "Low"
            badge_class = "badge-low-risk"
            badge_icon = "✅"

        # -----------------------------
        # Identify key risk factors
        # -----------------------------
        risk_factors = []

        if overtime == "Yes":
            risk_factors.append("Frequent overtime")

        if total_satisfaction <= 2.5:
            risk_factors.append("Low overall satisfaction")

        if years_since_promotion >= 5:
            risk_factors.append("Long time since last promotion")

        if monthly_income < 4000:
            risk_factors.append("Relatively low monthly income")

        if years_at_company >= 10:
            risk_factors.append("Long organizational tenure")

        if distance_from_home >= 20:
            risk_factors.append("Long commute distance")

        if business_travel == "Travel_Frequently":
            risk_factors.append("Frequent business travel")

        if stock_option == 0:
            risk_factors.append("No stock options")

        if len(risk_factors) == 0:
            risk_factors.append("No major operational risk indicators detected")

        with st.container(border=True):
            st.markdown(
                '<div class="result-title">📈 Attrition Prediction Report</div>',
                unsafe_allow_html=True
            )

            c1, c2 = st.columns([1, 2])

            with c1:

                st.metric(
                    "Attrition Probability",
                    f"{prob*100:.1f}%"
                )

                st.markdown(
                    f"""
                    <div class="result-badge {badge_class}">
                        {badge_icon} <b>{risk_level} Risk</b><br>
                        Decision Threshold : {threshold*100:.1f}%
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with c2:

                st.markdown(
                    """
                    <div class="result-title" style="margin-top:0;">
                    🔍 Key Risk Factors
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                for factor in risk_factors:
                    st.markdown(f"- {factor}")

            st.markdown("---")

            if risk_level == "High":

                st.error(
                    "This employee exhibits a **high likelihood of attrition** based on the operational features entered."
                )

                recommendations = []

                if overtime == "Yes":
                    recommendations.append(
                        "Reduce overtime workload or provide compensatory benefits."
                    )

                if total_satisfaction <= 2.5:
                    recommendations.append(
                        "Conduct engagement survey and manager 1-on-1 discussion."
                    )

                if years_since_promotion >= 5:
                    recommendations.append(
                        "Review promotion eligibility and career progression."
                    )

                if monthly_income < 4000:
                    recommendations.append(
                        "Benchmark salary against market standards."
                    )

                if distance_from_home >= 20:
                    recommendations.append(
                        "Consider hybrid/remote work options."
                    )

                if business_travel == "Travel_Frequently":
                    recommendations.append(
                        "Review travel frequency to reduce burnout."
                    )

                if stock_option == 0:
                    recommendations.append(
                        "Evaluate retention incentives such as stock options or bonuses."
                    )

            elif risk_level == "Moderate":

                st.warning(
                    "This employee shows a **moderate attrition risk**. Preventive HR intervention is recommended."
                )

                recommendations = [
                    "Schedule periodic engagement discussions.",
                    "Monitor workload and work-life balance.",
                    "Review career development opportunities.",
                    "Track employee satisfaction in upcoming reviews."
                ]

            else:

                st.success(
                    "This employee currently shows a **low probability of attrition**."
                )

                recommendations = [
                    "Continue regular employee engagement practices.",
                    "Maintain competitive compensation.",
                    "Monitor satisfaction periodically.",
                    "Recognize employee contributions to sustain retention."
                ]

            st.markdown(
                """
                <div class="result-title" style="margin-top:25px;">
                💡 Recommended HR Actions
                </div>
                """,
                unsafe_allow_html=True
            )

            for rec in recommendations:
                st.markdown(f"✅ {rec}")

            st.markdown("---")

            st.info(
                f"""
**Model Decision Summary**

- Predicted Probability : **{prob*100:.1f}%**
- Classification Threshold : **{threshold*100:.1f}%**
- Final Prediction : **{'Employee Likely to Leave' if is_attrition else 'Employee Likely to Stay'}**

This prediction is generated using the trained Logistic Regression model and should be used as a decision-support tool rather than a replacement for managerial judgment.
"""
            )

# ----------------------------------------------------
# TAB 2: GLASSDOOR REVIEW SENTIMENT ANALYZER
# ----------------------------------------------------
with tab2:

    st.markdown('<div class="card-label">📂 About the Data</div>', unsafe_allow_html=True)
    st.markdown("""
    <p class="section-desc">
    This model is trained on real Glassdoor employee reviews, scoped to Tech, Consulting,
    and Finance firms. Retail, food service, and hospitality employers were excluded since
    they represent a structurally different employee population than the salaried,
    corporate roles this project focuses on.
    </p>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.markdown(f"""
            <div class="kpi-card kpi-neutral">
                <div class="kpi-label">Reviews Analyzed</div>
                <div class="kpi-value">{config['n_total_sample']:,}</div>
            </div>
            """, unsafe_allow_html=True)
        with d2:
            st.markdown(f"""
            <div class="kpi-card kpi-neutral">
                <div class="kpi-label">Companies Covered</div>
                <div class="kpi-value">{len(config['allowed_firms'])}</div>
            </div>
            """, unsafe_allow_html=True)
        with d3:
            st.markdown("""
            <div class="kpi-card kpi-positive">
                <div class="kpi-label">Positive Reviews</div>
                <div class="kpi-value">84%</div>
            </div>
            """, unsafe_allow_html=True)
        with d4:
            st.markdown("""
            <div class="kpi-card kpi-negative">
                <div class="kpi-label">Negative Reviews</div>
                <div class="kpi-value">16%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="font-size: 0.78rem; color: #64748b; margin-top: 14px; line-height: 1.5;">
        Note: large, well-known employers tend to receive mostly positive reviews on
        Glassdoor — this 84:16 split reflects that real-world pattern, not a data error.
        Neutral 3-star reviews were excluded so the model learns from clearer positive
        and negative signal.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="card-label">⚙️ What We Did</div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("""
        <div style="font-size: 0.95rem; line-height: 1.8; color: #cbd5e1;">
        Each review's "pros" and "cons" text was cleaned and combined into a single
        document. We then converted that text into <strong style="color:#a5b4fc;">TF-IDF
        vectors</strong> — numeric scores that capture which words and short phrases are
        most distinctive to a review, automatically down-weighting common, uninformative
        words like "the" or "and". A <strong style="color:#a5b4fc;">Logistic Regression</strong>
        classifier was then trained on these vectors to predict whether a review reflects
        positive or negative sentiment. The model was validated using
        <strong style="color:#a5b4fc;">5-fold cross-validation</strong> and evaluated on a
        held-out test set it never saw during training, to give an honest measure of
        real-world performance.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="card-label">📈 Results</div>', unsafe_allow_html=True)

    with st.container(border=True):
        r1, r2 = st.columns(2)
        with r1:
            st.markdown(f"""
            <div class="kpi-card kpi-neutral">
                <div class="kpi-label">Test AUC-ROC</div>
                <div class="kpi-value">{config['test_auc_roc']:.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div class="kpi-card kpi-neutral">
                <div class="kpi-label">Cross-Validation AUC</div>
                <div class="kpi-value">{config['cv_auc_mean']:.3f} ± {config['cv_auc_std']:.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size: 0.78rem; color: #64748b; margin-top: 14px;">
        AUC-ROC measures how well the model separates positive from negative reviews;
        1.0 is perfect, 0.5 is random guessing.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="card-label">🔑 Top Words Driving Predictions</div>', unsafe_allow_html=True)

        top_pos = sorted(
            [(w, c) for w, c in feature_coef_map.items() if c > 0],
            key=lambda x: x[1], reverse=True
        )[:10]
        top_neg = sorted(
            [(w, c) for w, c in feature_coef_map.items() if c < 0],
            key=lambda x: x[1]
        )[:10]

        chart_df = pd.DataFrame(
            [{"word": w, "weight": c, "type": "Positive"} for w, c in top_pos] +
            [{"word": w, "weight": c, "type": "Negative"} for w, c in top_neg]
        ).sort_values("weight")

        fig = px.bar(
            chart_df, x="weight", y="word", color="type", orientation="h",
            color_discrete_map={"Positive": "#34d399", "Negative": "#f87171"},
            labels={"weight": "Model Coefficient", "word": ""},
            height=520
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#cbd5e1", legend_title_text="",
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="card-label">🏆 Firm Sentiment Leaderboard</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size: 0.78rem; color: #64748b; margin-bottom: 10px;">
        Based only on test-set reviews the model never trained on.
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(
            firm_leaderboard.rename(columns={
                "firm": "Company / Firm Name",
                "n_reviews": "Total Reviews",
                "pct_positive": "% Positive Predicted Sentiment",
                "avg_pos_prob": "Avg Positive Probability Score",
                "avg_rating": "Average Star Rating"
            }),
            use_container_width=True,
            hide_index=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="card-label">🔮 Try It Live</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-desc">Paste a custom company review written by an employee to analyze its sentiment polarity and identify specific key triggers.</p>', unsafe_allow_html=True)

    col_input, col_result = st.columns([3, 2])

    with col_input:
        with st.container(border=True):
            st.markdown('<div class="card-label">✍️ Analyze Custom Review</div>', unsafe_allow_html=True)

            review_text = st.text_area("Write or paste employee review here:", height=200,
                                       placeholder="Type something like: The work-life balance is great, but management is terrible and very toxic. Pay is okay but promotions are slow.",
                                       label_visibility="collapsed")

            analyze_btn = st.button("🔍  Evaluate Sentiment", use_container_width=True)

    with col_result:
        with st.container(border=True):
            st.markdown('<div class="card-label">📊 Prediction Result</div>', unsafe_allow_html=True)

            if analyze_btn and review_text.strip():
                clean_text = review_text.lower().strip()
                prob = nlp_pipeline.predict_proba([clean_text])[0, 1]
                pred_label = 1 if prob >= 0.50 else 0

                if pred_label == 1:
                    st.markdown('<div class="result-badge badge-positive" style="font-size: 1.15rem;">✅ Positive Sentiment</div>', unsafe_allow_html=True)
                    confidence = prob * 100
                else:
                    st.markdown('<div class="result-badge badge-negative" style="font-size: 1.15rem;">⚠️ Negative Sentiment</div>', unsafe_allow_html=True)
                    confidence = (1 - prob) * 100

                st.write("")
                st.metric("Confidence Score", f"{confidence:.1f}%")

                st.markdown('<div class="card-label" style="margin-top: 1.2rem; font-size: 0.7rem;">🔑 Highlighted Keywords</div>', unsafe_allow_html=True)
                highlighted_html = get_html_highlighted_text(review_text, feature_coef_map, threshold=0.15)

                st.markdown(f"""
                <div class="highlight-box">
                    {highlighted_html}
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div style="font-size: 0.78rem; color: #64748b; margin-top: 10px; line-height: 1.5;">
                    <span style="color: #86efac; font-weight: 600;">● Green</span> = positive triggers &nbsp;&nbsp;
                    <span style="color: #fca5a5; font-weight: 600;">● Red</span> = negative triggers
                </div>
                """, unsafe_allow_html=True)

            elif analyze_btn:
                st.warning("Please enter some review text to analyze.")
            else:
                st.markdown("""
                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 3rem 1rem; color: #475569; text-align: center;">
                    <div style="font-size: 3rem; margin-bottom: 1rem; opacity: 0.5;">💬</div>
                    <div style="font-size: 0.95rem; line-height: 1.6;">Input a review on the left and click<br><strong style="color: #818cf8;">'Evaluate Sentiment'</strong> to see results.</div>
                </div>
                """, unsafe_allow_html=True)