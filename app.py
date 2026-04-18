import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import os
import joblib
warnings.filterwarnings("ignore")

# ─────────────────────────── Page config ────────────────────────────
st.set_page_config(
    page_title="🚦 Road Accident Risk Predictor",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── Custom CSS ─────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@400;500;600&display=swap');

    :root {
        --accent:   #FF4B4B;
        --accent2:  #4361EE;
        --accent3:  #3A0CA3;
        --light-bg: #F8F9FA;
        --card-bg:  rgba(255, 255, 255, 0.85);
        --border:   rgba(255, 255, 255, 0.4);
        --text:     #2B2D42;
        --muted:    #8D99AE;
    }

    body, .stApp {
        background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%) !important;
        font-family: 'Inter', sans-serif !important;
        color: var(--text);
    }

    h1, h2, h3, .kpi-value, .kpi-label, .pred-box {
        font-family: 'Outfit', sans-serif !important;
    }

    /* ── Hero banner ── */
    .hero {
        background: linear-gradient(135deg, #3A0CA3 0%, #4361EE 50%, #4CC9F0 100%);
        border-radius: 24px;
        padding: 3rem 2.5rem;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 15px 35px rgba(67, 97, 238, 0.25);
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: ''; position: absolute; top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 60%);
        pointer-events: none;
    }
    .hero h1 { font-size: 3rem; font-weight: 800; margin: 0 0 .5rem; letter-spacing: -1px; text-shadow: 0 2px 10px rgba(0,0,0,0.15); }
    .hero p  { font-size: 1.15rem; opacity: .9; margin: 0; font-weight: 300;}

    /* ── KPI cards (Glassmorphism) ── */
    .kpi-row { display: flex; gap: 1.2rem; margin-bottom: 1.8rem; flex-wrap: wrap; }
    .kpi-card {
        flex: 1; min-width: 160px;
        background: var(--card-bg);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid var(--border);
        border-bottom: 4px solid var(--accent2);
        box-shadow: 0 8px 32px rgba(0,0,0,.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover { transform: translateY(-5px); box-shadow: 0 12px 40px rgba(0,0,0,.08); }
    .kpi-card.red   { border-bottom-color: #F72585; }
    .kpi-card.green { border-bottom-color: #4CC9F0; }
    .kpi-card.gold  { border-bottom-color: #F8961E; }
    .kpi-card.blue  { border-bottom-color: #4361EE; }
    
    .kpi-label  { font-size: .8rem; font-weight: 600; text-transform: uppercase;
                  letter-spacing: .1em; color: var(--muted); margin-bottom: .4rem; }
    .kpi-value  { font-size: 2.2rem; font-weight: 800; color: #3A0CA3; line-height: 1; }
    .kpi-sub    { font-size: .8rem; color: var(--muted); margin-top: .4rem; }

    /* ── Section headers ── */
    .section-header {
        font-family: 'Outfit', sans-serif;
        font-size: 1.5rem; font-weight: 800; color: #3A0CA3;
        padding-bottom: .6rem; margin: 2rem 0 1.2rem;
        position: relative;
    }
    .section-header::after {
        content: ''; position: absolute; bottom: 0; left: 0;
        width: 60px; height: 4px; background: #4CC9F0; border-radius: 2px;
    }

    /* ── Prediction result ── */
    .pred-box {
        border-radius: 20px;
        padding: 2.5rem 2rem;
        text-align: center;
        font-weight: 800;
        font-size: 1.2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,.08);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.6);
        transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .pred-box:hover { transform: scale(1.02); }
    .pred-low    { background: linear-gradient(135deg, rgba(216,243,220,0.9), rgba(82,183,136,0.2)); color: #1b4332; }
    .pred-medium { background: linear-gradient(135deg, rgba(255,243,176,0.9), rgba(233,196,106,0.2)); color: #7d4e00; }
    .pred-high   { background: linear-gradient(135deg, rgba(255,224,212,0.9), rgba(230,57,70,0.2)); color: #6d1a0e; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.6) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] h2 { color: #3A0CA3 !important; font-family: 'Outfit', sans-serif; font-weight: 800;}
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stCheckbox label { color: #2B2D42 !important; font-weight: 600; font-size: .85rem; }
    [data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #F72585 0%, #4361EE 100%) !important;
        color: white !important; border: none;
        border-radius: 12px; font-weight: 800; width: 100%; margin-top: 1rem;
        padding: 0.8rem 0; font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(67, 97, 238, 0.3);
        transition: all 0.2s ease;
    }
    [data-testid="stSidebar"] .stButton > button:hover { 
        box-shadow: 0 6px 20px rgba(247, 37, 133, 0.4); 
        transform: translateY(-2px);
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] { background: transparent; gap: 8px;}
    .stTabs [data-baseweb="tab"] { font-weight: 600; font-size: 1rem; font-family: 'Outfit', sans-serif; padding: 10px 20px; border-radius: 8px 8px 0 0;}
    .stTabs [aria-selected="true"] { background: var(--card-bg) !important; color: #3A0CA3 !important; border-bottom-color: #3A0CA3 !important; border-bottom-width: 3px !important; }

    /* ── Misc ── */
    .stDataFrame { border-radius: 16px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,.05); border: 1px solid var(--border); }
    div[data-testid="stMetricValue"] { font-size: 1.8rem !important; font-family: 'Outfit', sans-serif; font-weight: 800; color: #3A0CA3;}
    .footer { text-align:center; color:#8D99AE; font-size:.9rem; padding:2.5rem 0 1rem; font-weight: 500;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── Data loading ───────────────────────────
@st.cache_data(show_spinner=False)
def load_data(path):
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def preprocess(df):
    df = df.copy()
    cat_cols = ["road_type", "lighting", "weather", "time_of_day"]
    bool_cols = ["road_signs_present", "public_road", "holiday", "school_season"]
    le = {}
    for c in cat_cols:
        enc = LabelEncoder()
        df[c] = enc.fit_transform(df[c].astype(str))
        le[c] = enc
    for c in bool_cols:
        df[c] = df[c].astype(int)
    return df, le

@st.cache_resource(show_spinner=False)
def train_models(X_tr, y_tr):
    models = {
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.08,
                                                        max_depth=5, random_state=42),
        "Random Forest":     RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1),
        "Ridge Regression":  Ridge(alpha=1.0),
    }
    trained = {}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        trained[name] = m
    return trained

# ────────────────────────── Load & process ──────────────────────────
with st.spinner("Loading data…"):
    DATA_PATH = os.path.join(os.path.dirname(__file__), "train.csv")
    raw = load_data(DATA_PATH)

FEATURE_COLS = ["road_type","num_lanes","curvature","speed_limit","lighting",
                "weather","road_signs_present","public_road","time_of_day",
                "holiday","school_season","num_reported_accidents"]
TARGET = "accident_risk"

proc_df, label_enc = preprocess(raw)
X = proc_df[FEATURE_COLS].values
y = proc_df[TARGET].values

CACHE_FILE = os.path.join(os.path.dirname(__file__), "model_artifacts.joblib")

if os.path.exists(CACHE_FILE):
    artifacts = joblib.load(CACHE_FILE)
    scaler = artifacts["scaler"]
    trained_models = artifacts["trained_models"]

    _, X_val, _, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
    X_val_sc = scaler.transform(X_val)

    # Re-compute predictions so they match the current y_val exactly
    eval_results = {}
    for name, m in trained_models.items():
        preds = m.predict(X_val_sc)
        eval_results[name] = {
            "RMSE": np.sqrt(mean_squared_error(y_val, preds)),
            "MAE":  mean_absolute_error(y_val, preds),
            "R²":   r2_score(y_val, preds),
            "preds": preds,
        }
else:
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
    
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)

    with st.spinner("Training models for the first time… ☕"):
        trained_models = train_models(X_train_sc, y_train)

    # Evaluate all models
    eval_results = {}
    for name, m in trained_models.items():
        preds = m.predict(X_val_sc)
        eval_results[name] = {
            "RMSE": np.sqrt(mean_squared_error(y_val, preds)),
            "MAE":  mean_absolute_error(y_val, preds),
            "R²":   r2_score(y_val, preds),
            "preds": preds,
        }
    
    # Save artifacts for future runs
    joblib.dump({"scaler": scaler, "trained_models": trained_models, "eval_results": eval_results}, CACHE_FILE)

best_model_name = min(eval_results, key=lambda k: eval_results[k]["RMSE"])
best_model      = trained_models[best_model_name]

# ─────────────────────────── Sidebar ────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Predict New Road")
    st.markdown("---")
    road_type   = st.selectbox("Road Type",      ["urban","rural","highway"])
    num_lanes   = st.slider("Number of Lanes",   1, 6, 2)
    curvature   = st.slider("Curvature",         0.0, 1.0, 0.2, 0.01)
    speed_limit = st.selectbox("Speed Limit",    [25, 35, 45, 60, 70])
    lighting    = st.selectbox("Lighting",       ["daylight","dim","night"])
    weather     = st.selectbox("Weather",        ["clear","rainy","foggy"])
    time_of_day = st.selectbox("Time of Day",    ["morning","afternoon","evening"])
    road_signs  = st.checkbox("Road Signs Present", value=True)
    public_road = st.checkbox("Public Road",        value=True)
    holiday     = st.checkbox("Holiday",            value=False)
    school_season=st.checkbox("School Season",      value=False)
    num_acc     = st.slider("Past Reported Accidents", 0, 10, 1)
    chosen_model= st.selectbox("Model", list(trained_models.keys()), index=list(trained_models.keys()).index(best_model_name))
    predict_btn = st.button("🔍 Predict Risk")

# ─────────────────────────── Hero ───────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🚦 Road Accident Risk Predictor</h1>
  <p>Machine-learning powered analysis of road safety factors &bull;
     Gradient Boosting · Random Forest · Ridge Regression</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────── KPI bar ────────────────────────────────
best = eval_results[best_model_name]
st.markdown(f"""
<div class="kpi-row">
  <div class="kpi-card blue">
    <div class="kpi-label">Training Rows</div>
    <div class="kpi-value">{len(raw):,}</div>
    <div class="kpi-sub">road segments</div>
  </div>
  <div class="kpi-card green">
    <div class="kpi-label">Best Model</div>
    <div class="kpi-value" style="font-size:1.15rem;padding-top:.35rem">{best_model_name}</div>
    <div class="kpi-sub">lowest RMSE</div>
  </div>
  <div class="kpi-card red">
    <div class="kpi-label">Best RMSE</div>
    <div class="kpi-value">{best['RMSE']:.4f}</div>
    <div class="kpi-sub">validation set</div>
  </div>
  <div class="kpi-card gold">
    <div class="kpi-label">Best R²</div>
    <div class="kpi-value">{best['R²']:.4f}</div>
    <div class="kpi-sub">explained variance</div>
  </div>
  <div class="kpi-card blue">
    <div class="kpi-label">Features Used</div>
    <div class="kpi-value">{len(FEATURE_COLS)}</div>
    <div class="kpi-sub">road attributes</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────── Prediction panel ───────────────────────
if predict_btn:
    rt_enc  = label_enc["road_type"].transform([road_type])[0]
    li_enc  = label_enc["lighting"].transform([lighting])[0]
    we_enc  = label_enc["weather"].transform([weather])[0]
    tod_enc = label_enc["time_of_day"].transform([time_of_day])[0]
    inp = np.array([[rt_enc, num_lanes, curvature, speed_limit, li_enc,
                     we_enc, int(road_signs), int(public_road),
                     tod_enc, int(holiday), int(school_season), num_acc]])
    inp_sc = scaler.transform(inp)
    pred   = float(trained_models[chosen_model].predict(inp_sc)[0])
    pred   = np.clip(pred, 0, 1)

    if pred < 0.30:
        box_cls, emoji, label = "pred-low",    "✅", "LOW RISK"
    elif pred < 0.60:
        box_cls, emoji, label = "pred-medium", "⚠️", "MEDIUM RISK"
    else:
        box_cls, emoji, label = "pred-high",   "🚨", "HIGH RISK"

    st.markdown("<div class='section-header'>🎯 Prediction Result</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.markdown(f"""
        <div class="pred-box {box_cls}">
          {emoji} {label}<br>
          <span style="font-size:3rem;font-weight:900;">{pred:.3f}</span><br>
          <span style="font-weight:400;font-size:.95rem;">Accident Risk Score (0 – 1) using {chosen_model}</span>
        </div>""", unsafe_allow_html=True)

    # gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred,
        number={"font": {"size": 48, "color": "#1d3557"}, "suffix": ""},
        gauge={
            "axis": {"range": [0, 1], "tickwidth": 1, "tickcolor": "#1d3557"},
            "bar":  {"color": "#e63946" if pred >= 0.6 else ("#f4a261" if pred >= 0.3 else "#2a9d8f"), "thickness": 0.25},
            "bgcolor": "white",
            "bordercolor": "#dee2e6",
            "steps": [
                {"range": [0, 0.3], "color": "#d8f3dc"},
                {"range": [0.3, 0.6], "color": "#fff3b0"},
                {"range": [0.6, 1.0], "color": "#ffe0d4"},
            ],
            "threshold": {"line": {"color": "#1d3557", "width": 3}, "thickness": 0.75, "value": pred},
        },
        title={"text": "Risk Gauge", "font": {"size": 16, "color": "#1d3557"}},
    ))
    fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10),
                             paper_bgcolor="white", font={"color": "#1d3557"})
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.plotly_chart(fig_gauge, use_container_width=True)

# ─────────────────────────── Tabs ───────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", " Model Performance", "📈 Feature Importance", "🗃️ Data Preview"])

# ── Tab 1 : EDA ──────────────────────────────────────────────────────
with tab1:
    st.markdown("<div class='section-header'>Exploratory Data Analysis</div>", unsafe_allow_html=True)
    sample = raw.sample(min(20000, len(raw)), random_state=42)

    col1, col2 = st.columns(2)

    # Target distribution
    with col1:
        fig = px.histogram(sample, x="accident_risk", nbins=50,
                           color_discrete_sequence=["#457b9d"],
                           title="Distribution of Accident Risk",
                           labels={"accident_risk": "Accident Risk Score"},
                           template="plotly_white")
        fig.add_vline(x=sample["accident_risk"].mean(), line_dash="dash",
                      line_color="#e63946", annotation_text=f"Mean: {sample['accident_risk'].mean():.3f}")
        fig.update_layout(bargap=0.05, showlegend=False,
                          title_font_size=15, title_font_color="#1d3557")
        st.plotly_chart(fig, use_container_width=True)

    # Box by road type
    with col2:
        fig = px.box(sample, x="road_type", y="accident_risk",
                     color="road_type",
                     color_discrete_sequence=["#e63946","#457b9d","#2a9d8f"],
                     title="Risk by Road Type",
                     template="plotly_white",
                     labels={"road_type":"Road Type","accident_risk":"Accident Risk"})
        fig.update_layout(showlegend=False, title_font_size=15, title_font_color="#1d3557")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    # Risk by weather
    with col3:
        means = sample.groupby("weather")["accident_risk"].mean().reset_index().sort_values("accident_risk", ascending=False)
        fig = px.bar(means, x="weather", y="accident_risk",
                     color="weather",
                     color_discrete_sequence=["#a8dadc","#457b9d","#1d3557"],
                     title="Avg Risk by Weather Condition",
                     template="plotly_white",
                     text_auto=".3f",
                     labels={"weather":"Weather","accident_risk":"Mean Risk"})
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, title_font_size=15, title_font_color="#1d3557")
        st.plotly_chart(fig, use_container_width=True)

    # Risk by time of day
    with col4:
        means2 = sample.groupby("time_of_day")["accident_risk"].mean().reset_index()
        fig = px.bar(means2, x="time_of_day", y="accident_risk",
                     color="time_of_day",
                     color_discrete_sequence=["#f4a261","#e76f51","#264653"],
                     title="Avg Risk by Time of Day",
                     template="plotly_white",
                     text_auto=".3f",
                     labels={"time_of_day":"Time","accident_risk":"Mean Risk"})
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, title_font_size=15, title_font_color="#1d3557")
        st.plotly_chart(fig, use_container_width=True)

    col5, col6 = st.columns(2)

    # Scatter: curvature vs risk
    with col5:
        scatter_s = sample.sample(3000, random_state=1)
        fig = px.scatter(scatter_s, x="curvature", y="accident_risk",
                         color="road_type",
                         color_discrete_sequence=["#e63946","#457b9d","#2a9d8f"],
                         opacity=0.45,
                         title="Curvature vs Accident Risk",
                         template="plotly_white",
                         trendline="lowess",
                         labels={"curvature":"Road Curvature","accident_risk":"Risk"})
        fig.update_layout(title_font_size=15, title_font_color="#1d3557")
        st.plotly_chart(fig, use_container_width=True)

    # Heat: lighting × weather
    with col6:
        heat = sample.groupby(["lighting","weather"])["accident_risk"].mean().unstack()
        fig = px.imshow(heat, text_auto=".3f",
                        color_continuous_scale=["#d8f3dc","#fff3b0","#e63946"],
                        title="Avg Risk: Lighting × Weather",
                        template="plotly_white",
                        labels={"color":"Mean Risk"})
        fig.update_layout(title_font_size=15, title_font_color="#1d3557")
        st.plotly_chart(fig, use_container_width=True)

    # Violin: lanes
    fig = px.violin(sample, x="num_lanes", y="accident_risk",
                    color="num_lanes", box=True, points=False,
                    title="Risk Distribution by Number of Lanes",
                    template="plotly_white",
                    color_discrete_sequence=px.colors.sequential.Blues_r,
                    labels={"num_lanes":"Lanes","accident_risk":"Accident Risk"})
    fig.update_layout(showlegend=False, title_font_size=15, title_font_color="#1d3557")
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 2 : Model Performance ────────────────────────────────────────
with tab2:
    st.markdown("<div class='section-header'>Model Performance Comparison</div>", unsafe_allow_html=True)

    perf_df = pd.DataFrame({k: {m: eval_results[m][k] for m in eval_results}
                             for k in ["RMSE","MAE","R²"]}).T
    perf_df.index.name = "Metric"

    # Metric comparison bar
    fig = make_subplots(rows=1, cols=3, subplot_titles=["RMSE ↓", "MAE ↓", "R² ↑"])
    colors = ["#e63946","#457b9d","#2a9d8f"]
    models_list = list(eval_results.keys())

    for col_i, metric in enumerate(["RMSE","MAE","R²"], 1):
        vals = [eval_results[m][metric] for m in models_list]
        fig.add_trace(go.Bar(x=models_list, y=vals, marker_color=colors,
                             text=[f"{v:.4f}" for v in vals], textposition="outside",
                             showlegend=False), row=1, col=col_i)

    fig.update_layout(title="Model Metrics on Validation Set",
                      template="plotly_white", height=380,
                      title_font_size=15, title_font_color="#1d3557")
    st.plotly_chart(fig, use_container_width=True)

    # Actual vs Predicted scatter for each model
    col1, col2, col3 = st.columns(3)
    for col_obj, name in zip([col1, col2, col3], models_list):
        with col_obj:
            preds = eval_results[name]["preds"]
            sample_idx = np.random.choice(len(y_val), 1000, replace=False)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_val[sample_idx], y=preds[sample_idx],
                                     mode="markers",
                                     marker=dict(color="#457b9d", opacity=0.4, size=4),
                                     name="Predictions"))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                     line=dict(color="#e63946", dash="dash", width=2),
                                     name="Perfect"))
            fig.update_layout(title=f"{name}<br><sup>R²={eval_results[name]['R²']:.4f}</sup>",
                               xaxis_title="Actual", yaxis_title="Predicted",
                               template="plotly_white", height=320,
                               showlegend=False, title_font_color="#1d3557")
            st.plotly_chart(fig, use_container_width=True)

    # Residuals
    st.markdown("<div class='section-header'>Residual Analysis — Best Model</div>", unsafe_allow_html=True)
    best_preds = eval_results[best_model_name]["preds"]
    residuals  = y_val - best_preds
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(residuals, nbins=60,
                           title=f"Residuals: {best_model_name}",
                           labels={"value":"Residual","count":"Frequency"},
                           color_discrete_sequence=["#2a9d8f"],
                           template="plotly_white")
        fig.add_vline(x=0, line_dash="dash", line_color="#e63946")
        fig.update_layout(title_font_color="#1d3557", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        sample_idx2 = np.random.choice(len(y_val), 2000, replace=False)
        fig = px.scatter(x=best_preds[sample_idx2], y=residuals[sample_idx2],
                         title="Predicted vs Residual",
                         labels={"x":"Predicted","y":"Residual"},
                         color_discrete_sequence=["#457b9d"],
                         opacity=0.5, template="plotly_white")
        fig.add_hline(y=0, line_dash="dash", line_color="#e63946")
        fig.update_layout(title_font_color="#1d3557")
        st.plotly_chart(fig, use_container_width=True)

# ── Tab 3 : Feature Importance ───────────────────────────────────────
with tab3:
    st.markdown("<div class='section-header'>Feature Importance</div>", unsafe_allow_html=True)

    gb_model = trained_models["Gradient Boosting"]
    rf_model = trained_models["Random Forest"]

    feat_names = ["road_type","num_lanes","curvature","speed_limit","lighting",
                  "weather","road_signs_present","public_road","time_of_day",
                  "holiday","school_season","num_reported_accidents"]

    gb_imp = pd.DataFrame({"Feature": feat_names, "Importance": gb_model.feature_importances_,
                            "Model": "Gradient Boosting"})
    rf_imp = pd.DataFrame({"Feature": feat_names, "Importance": rf_model.feature_importances_,
                            "Model": "Random Forest"})

    col1, col2 = st.columns(2)
    for col_obj, imp_df, title, color in [
        (col1, gb_imp, "Gradient Boosting Feature Importance", "#e63946"),
        (col2, rf_imp, "Random Forest Feature Importance",     "#457b9d"),
    ]:
        with col_obj:
            imp_sorted = imp_df.sort_values("Importance", ascending=True)
            fig = go.Figure(go.Bar(
                x=imp_sorted["Importance"], y=imp_sorted["Feature"],
                orientation="h",
                marker=dict(color=imp_sorted["Importance"],
                            colorscale=[[0,"#f0f4f8"],[1, color]],
                            showscale=False),
                text=[f"{v:.4f}" for v in imp_sorted["Importance"]],
                textposition="outside",
            ))
            fig.update_layout(title=title, template="plotly_white",
                              height=420, xaxis_title="Importance",
                              title_font_size=14, title_font_color="#1d3557",
                              margin=dict(l=10, r=60))
            st.plotly_chart(fig, use_container_width=True)

    # Cumulative importance
    gb_sorted = gb_imp.sort_values("Importance", ascending=False).reset_index(drop=True)
    gb_sorted["Cumulative"] = gb_sorted["Importance"].cumsum()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=gb_sorted["Feature"], y=gb_sorted["Importance"],
                         marker_color="#457b9d", name="Importance", opacity=0.8))
    fig.add_trace(go.Scatter(x=gb_sorted["Feature"], y=gb_sorted["Cumulative"],
                             mode="lines+markers", yaxis="y2",
                             line=dict(color="#e63946", width=2.5),
                             marker=dict(size=7), name="Cumulative"))
    fig.update_layout(
        title="Cumulative Feature Importance (Gradient Boosting)",
        template="plotly_white", height=400,
        yaxis2=dict(overlaying="y", side="right", range=[0, 1.05],
                    title="Cumulative Importance", tickformat=".0%"),
        yaxis=dict(title="Importance"),
        legend=dict(orientation="h", y=1.12),
        title_font_color="#1d3557",
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 4 : Data Preview ─────────────────────────────────────────────
with tab4:
    st.markdown("<div class='section-header'>Dataset Overview</div>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows",    f"{len(raw):,}")
    col2.metric("Total Columns", f"{raw.shape[1]}")
    col3.metric("Target Mean",   f"{raw['accident_risk'].mean():.4f}")
    col4.metric("Target Std",    f"{raw['accident_risk'].std():.4f}")

    st.markdown("**Sample (500 rows)**")
    st.dataframe(raw.sample(500, random_state=0).reset_index(drop=True), height=340, use_container_width=True)

    st.markdown("**Descriptive Statistics**")
    st.dataframe(raw.describe().T.style.background_gradient(cmap="Blues", subset=["mean","std"]),
                 use_container_width=True)

    # Correlation heatmap
    st.markdown("<div class='section-header'>Numeric Correlation Heatmap</div>", unsafe_allow_html=True)
    num_cols = ["num_lanes","curvature","speed_limit","num_reported_accidents","accident_risk"]
    corr = raw[num_cols].corr()
    fig = px.imshow(corr, text_auto=".2f",
                    color_continuous_scale=["#1d3557","white","#e63946"],
                    zmin=-1, zmax=1,
                    title="Pearson Correlation Matrix",
                    template="plotly_white")
    fig.update_layout(title_font_color="#1d3557", height=380)
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────── Footer ─────────────────────────────────
st.markdown("<div class='footer'>🚦 Road Accident Risk Predictor · Built with Streamlit & Scikit-learn</div>",
            unsafe_allow_html=True)
