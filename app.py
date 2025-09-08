# app.py
"""
Refactored EV Charging Infrastructure & GenAI Dashboard (Streamlit)
- Structured layout with Hero, KPIs, National trends, County hotspot, Forecast viewer,
  Scenario simulation, Charger type mix, AI Assistant, Executive summary.
- Groq LLM integration via call_genai_chat (select model in sidebar).
- GSAP animations in header (embedded HTML component). Has "Reduce motion" option.
- Save/load pickles, export CSVs, download charts as images (basic).
- Keep your forecasting/model code intact and integrated.
IMPORTANT:
 - Provide GROQ_API_KEY via environment variable GROQ_API_KEY or replace placeholder below for local testing.
 - Do NOT commit secrets to public repos.
"""

import os
import pickle
from datetime import datetime
import io
import base64
import time

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="EV Charging Infra + GenAI", layout="wide", initial_sidebar_state="expanded")

# --------------------------
# Load Environment Variables
# --------------------------
from dotenv import load_dotenv

GROQ_API_KEY = None

try:
    # First, try Streamlit Cloud secrets
    if "GROQ_API_KEY" in st.secrets:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    else:
        # Local development: load from grok.env or .env
        load_dotenv("grok.env")
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
except Exception as e:
    st.error(f"⚠️ Could not load GROQ_API_KEY: {e}")


# --------------------------
# Optional Lottie animation support
# --------------------------
try:
    from streamlit_lottie import st_lottie
    import requests  # used for fetching lottie json
    LOTTIE_AVAILABLE = True
except Exception:
    LOTTIE_AVAILABLE = False

# --------------------------
# Groq SDK
# --------------------------
try:
    from groq import Groq
    GROQ_SDK_AVAILABLE = True
except Exception:
    GROQ_SDK_AVAILABLE = False

# --------------------------
# Config / Keys
# --------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # only from env, no fallback
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

# --------------------------
# Init Groq client safely
# --------------------------
groq_client = None
if GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        print(f"⚠️ Error initializing Groq client: {e}")
else:
    print("❌ GROQ_API_KEY not found. Please check your grok.env or .env file.")



# --------------------------
# GenAI wrapper
# --------------------------
def call_genai_chat(prompt: str, model: str = DEFAULT_GROQ_MODEL, max_tokens: int = 400, temperature: float = 0.0) -> str:
    """
    Call Groq chat completions and return text content.
    If groq_client is not initialized, return friendly message.
    """
    if groq_client is None:
        return ("⚠️ Groq client is not initialized. Set GROQ_API_KEY environment variable or replace the placeholder "
                "for local testing. LLM features will not work until a valid key is provided.")
    try:
        resp = groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert assistant for EV charging infrastructure planning."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        # Extract content with safe fallbacks
        try:
            return resp.choices[0].message.content
        except Exception:
            try:
                return resp.choices[0].message["content"]
            except Exception:
                return str(resp)
    except Exception as e:
        return f"⚠️ Error calling Groq API: {e}"

# --------------------------
# Utilities / I/O
# --------------------------
@st.cache_data
def load_csv_data(path="preprocessed_ev_data.csv"):
    if not os.path.exists(path):
        st.error(f"Dataset not found at: {path}")
        return None
    df = pd.read_csv(path)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    numeric_cols = [
        "Battery Electric Vehicles (BEVs)",
        "Plug-In Hybrid Electric Vehicles (PHEVs)",
        "Electric Vehicle (EV) Total",
        "Non-Electric Vehicle Total",
        "Total Vehicles",
        "Percent Electric Vehicles",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors='coerce').fillna(0)
    if 'county_encoded' not in df.columns:
        df['county_encoded'] = pd.factorize(df['County'].fillna('Unknown'))[0]
    if 'months_since_start' not in df.columns:
        df = df.sort_values(['County', 'Date'])
        df['months_since_start'] = df.groupby('County').cumcount()
    return df

@st.cache_data
def load_model(path="forecasting_ev_model.pkl"):
    if not os.path.exists(path):
        st.warning(f"Model file not found: {path}. Forecast disabled.")
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def save_pickle(obj, fname):
    with open(fname, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(fname):
    if not os.path.exists(fname):
        return None
    with open(fname, "rb") as f:
        return pickle.load(f)

def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')

# --------------------------
# Domain functions (same as your logic)
# --------------------------
def compute_ev_demand(df_local, bev_kwh_per_day=12, phev_kwh_per_day=4, days_per_month=30,
                      bev_share=None, phev_share=None):
    d = df_local.copy()
    if 'Battery Electric Vehicles (BEVs)' in d.columns and 'Plug-In Hybrid Electric Vehicles (PHEVs)' in d.columns:
        d['Monthly_kWh_Demand'] = (
            d['Battery Electric Vehicles (BEVs)'] * bev_kwh_per_day * days_per_month +
            d['Plug-In Hybrid Electric Vehicles (PHEVs)'] * phev_kwh_per_day * days_per_month
        )
    else:
        if bev_share is None or phev_share is None:
            bev_share = 0.7
            phev_share = 0.3
        d['BEVs_est'] = d['Electric Vehicle (EV) Total'] * bev_share
        d['PHEVs_est'] = d['Electric Vehicle (EV) Total'] * phev_share
        d['Monthly_kWh_Demand'] = (
            d['BEVs_est'] * bev_kwh_per_day * days_per_month +
            d['PHEVs_est'] * phev_kwh_per_day * days_per_month
        )
    d['Monthly_GWh_Demand'] = d['Monthly_kWh_Demand'] / 1e6
    return d

def compute_chargers_and_grid(df_local, charger_kw=50, utilization_diversity=0.5, avg_hours_per_day=6):
    d = df_local.copy()
    monthly_kwh_per_charger = charger_kw * avg_hours_per_day * 30
    d['Chargers_Required'] = (d['Monthly_kWh_Demand'] / monthly_kwh_per_charger).fillna(0)
    d['Chargers_Required_int'] = np.ceil(d['Chargers_Required']).astype(int)
    d['Grid_Load_MW'] = (d['Chargers_Required_int'] * charger_kw * utilization_diversity) / 1000.0
    return d

# --------------------------
# Load data & model
# --------------------------
df = load_csv_data("preprocessed_ev_data.csv")
model = load_model("forecasting_ev_model.pkl")

# --------------------------
# Streamlit layout & styling
# --------------------------


st.markdown("""
<style>
:root{
  --bg:#f5f5dc;
  --accent:#1b4332;
  --muted:#2f6f46;
  --card:#ffffff;
}
body { background-color: var(--bg); }
.kpi-card { background: var(--card); border-radius: 12px; padding: 18px; box-shadow: 0 6px 20px rgba(27,67,50,0.08); }
.section-title { font-size:20px; color:var(--accent); font-weight:700; margin-bottom:6px; }
.small-muted { color:#6b6b6b; font-size:13px; }
</style>
""", unsafe_allow_html=True)

# Session state for AI responses & reduction of motion
if 'last_ai' not in st.session_state:
    st.session_state['last_ai'] = ""
if 'reduce_motion' not in st.session_state:
    st.session_state['reduce_motion'] = False

# --------------------------
# Sidebar: controls, LLM options
# --------------------------
st.sidebar.title("Controls")
if df is None:
    st.sidebar.error("Dataset not found.")
else:
    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()
    date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
if 'State' in (df.columns if df is not None else []):
    state_list = ["All"] + sorted(df['State'].dropna().unique().tolist())
else:
    state_list = ["All"]
state_sel = st.sidebar.selectbox("State (optional)", state_list, index=0)

st.sidebar.subheader("Scenario Presets")
preset = st.sidebar.selectbox("Preset", ["Base", "Aggressive EV Uptake", "Conservative"])
if preset == "Base":
    bev_kwh_day_default = 12
    phev_kwh_day_default = 4
    charger_kw_default = 50
    diversity_default = 0.5
elif preset == "Aggressive EV Uptake":
    bev_kwh_day_default = 15
    phev_kwh_day_default = 5
    charger_kw_default = 50
    diversity_default = 0.6
else:
    bev_kwh_day_default = 10
    phev_kwh_day_default = 3
    charger_kw_default = 50
    diversity_default = 0.4

st.sidebar.subheader("Customize scenario")
bev_kwh_day = st.sidebar.number_input("BEV avg kWh/day", value=bev_kwh_day_default, min_value=5, max_value=60)
phev_kwh_day = st.sidebar.number_input("PHEV avg kWh/day", value=phev_kwh_day_default, min_value=1, max_value=30)
charger_kw = st.sidebar.number_input("Charger power (kW)", value=charger_kw_default, min_value=7, max_value=300)
diversity = st.sidebar.slider("Diversity factor (simultaneous use)", min_value=0.1, max_value=1.0, value=float(diversity_default), step=0.05)
avg_hours_per_day = st.sidebar.number_input("Avg charger usage (hrs/day)", value=6, min_value=1, max_value=24)

st.sidebar.markdown("---")
st.sidebar.subheader("GenAI (Groq) settings")
llm_model = st.sidebar.selectbox("Model", [DEFAULT_GROQ_MODEL, "gemma-7b", "gemma-7b-instruct"], index=0)
llm_temp = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0, step=0.05)
llm_max_tokens = st.sidebar.number_input("Max tokens", value=350, min_value=50, max_value=2000, step=50)

st.sidebar.markdown("---")
st.session_state['reduce_motion'] = st.sidebar.checkbox("Reduce motion (accessibility)", value=st.session_state['reduce_motion'])

st.sidebar.markdown("---")
if st.sidebar.button("Save computed DataFrames"):
    try:
        ev_demand_df_temp = compute_ev_demand(df, bev_kwh_day, phev_kwh_day)
        ev_demand_df_temp = compute_chargers_and_grid(ev_demand_df_temp, charger_kw, diversity, avg_hours_per_day)
        save_pickle(ev_demand_df_temp, "ev_demand_df.pkl")
        save_pickle(ev_demand_df_temp.groupby(['Date'])['Monthly_GWh_Demand'].sum().reset_index(), "infra_trend.pkl")
        county_demand_df = ev_demand_df_temp.groupby('County')[['Monthly_GWh_Demand', 'Chargers_Required_int', 'Grid_Load_MW']].sum().reset_index()
        county_demand_df = county_demand_df.rename(columns={'Chargers_Required_int': 'Chargers_Required'})
        save_pickle(county_demand_df, "county_demand.pkl")
        st.sidebar.success("Saved ev_demand_df.pkl, infra_trend.pkl, county_demand.pkl")
    except Exception as e:
        st.sidebar.error(f"Save error: {e}")

if st.sidebar.button("Load precomputed pickles"):
    ev_demand_df_loaded = load_pickle("ev_demand_df.pkl")
    county_demand_loaded = load_pickle("county_demand.pkl")
    infra_trend_loaded = load_pickle("infra_trend.pkl")
    if ev_demand_df_loaded is not None:
        st.sidebar.success("Loaded pickles.")
    else:
        st.sidebar.info("No pickles found; compute will run automatically.")

st.sidebar.markdown("---")
st.sidebar.caption("Data: Washington DOL (sample). Groq LLM used for AI text generation.")

# --------------------------
# Hero Section (Landing with Anime.js + Lottie)
# --------------------------
from streamlit.components.v1 import html as st_html

hero_html = f"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <!-- Anime.js -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/animejs/3.2.1/anime.min.js"></script>
    <!-- Lottie -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/lottie-web/5.12.2/lottie.min.js"></script>

    <style>
      body {{
        margin: 0; padding: 0;
        background: linear-gradient(135deg, #d8f3dc, #fffbe7, #caf0f8);
        font-family: Arial, Helvetica, sans-serif;
      }}
      .wrap {{
        display: flex; align-items: center; justify-content: space-between;
        gap: 40px; padding: 30px 40px;
      }}
      .left {{ flex: 1; }}
      .title {{
        font-size: 36px; font-weight: 800;
        color: #1b4332; margin: 0;
        line-height: 1.2;
      }}
      .title span {{ display: inline-block; }}
      .sub {{
        font-size: 16px; color: #2f6f46;
        margin-top: 14px; opacity: 0;
        text-shadow: 0 0 8px rgba(46, 204, 113, 0.4);
      }}
      .cta {{
        margin-top: 20px; display: flex; gap: 14px;
      }}
      .btn {{
        padding: 10px 16px; border-radius: 10px;
        background: #1b4332; color: white;
        cursor: pointer; font-weight: 700; position: relative;
        overflow: hidden; transition: transform 0.2s ease;
      }}
      .btn:hover {{ transform: scale(1.05); }}
      .btn:hover::after {{
        content: ""; position: absolute; top: 50%; left: 50%;
        width: 0; height: 0; background: rgba(255,255,255,0.3);
        border-radius: 50%; transform: translate(-50%, -50%);
        animation: ripple 0.6s linear;
      }}
      @keyframes ripple {{
        to {{ width: 220%; height: 220%; opacity: 0; }}
      }}
      .right {{ width: 340px; }}
      #lottie {{ width: 320px; height: 180px; }}
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="left">
        <div class="title" id="hero-title">
          EV Charging Infrastructure & Grid Impact Dashboard
        </div>
        <div class="sub" id="hero-sub">
          Interactive forecasting, county prioritization, scenario simulation and AI-generated recommendations.
        </div>
        <div class="cta">
          <div class="btn" id="demo-btn">⚡ Run Quick Demo</div>
          <div class="btn" id="explain-btn" style="background:#3a8d7a">🌍 Explain Forecast</div>
        </div>
      </div>
      <div class="right">
        <div id="lottie"></div>
      </div>
    </div>

    <script>
      const reduce = {str(st.session_state.get('reduce_motion', False)).lower()};
      if(!reduce) {{
        // --- Title Wave Animation ---
        let title = document.querySelector("#hero-title");
        title.innerHTML = title.textContent.replace(/(\S)/g, "<span>$1</span>");

        anime.timeline()
          .add({{
            targets: '#hero-title span',
            translateY: [50, 0],
            opacity: [0, 1],
            easing: "easeOutExpo",
            duration: 600,
            delay: anime.stagger(40)
          }});

        // --- Subtext Fade + Glow ---
        anime({{
          targets: '#hero-sub',
          opacity: [0, 1],
          duration: 800,
          delay: 700,
          easing: 'easeOutQuad'
        }});

        // --- Buttons Pulse ---
        anime({{
          targets: '#demo-btn',
          scale: [
            {{ value: 1.05, duration: 600, easing: "easeInOutSine" }},
            {{ value: 1.0, duration: 600, easing: "easeInOutSine" }}
          ],
          loop: true
        }});
        anime({{
          targets: '#explain-btn',
          scale: [
            {{ value: 1.08, duration: 800, easing: "easeInOutSine" }},
            {{ value: 1.0, duration: 800, easing: "easeInOutSine" }}
          ],
          loop: true
        }});
      }}

      // --- Lottie EV Illustration ---
      var animation = lottie.loadAnimation({{
        container: document.getElementById('lottie'),
        renderer: 'svg',
        loop: true,
        autoplay: true,
        path: 'https://assets7.lottiefiles.com/packages/lf20_tljjahas.json'
      }});

      // Send click messages to Streamlit
      document.getElementById("demo-btn").addEventListener("click", function(){{
        window.postMessage({{type: "demo_clicked"}}, "*");
      }});
      document.getElementById("explain-btn").addEventListener("click", function(){{
        window.postMessage({{type: "explain_clicked"}}, "*");
      }});
    </script>
  </body>
</html>
"""

st_html(hero_html, height=250)



# Listen to messages from component (simple polling via st experimental get_query_params not available)
# Instead use Streamlit buttons or rely on user flow - we'll not rely on postMessage here.

# --------------------------
# Compute core metrics
# --------------------------
if df is None:
    st.stop()

# Filter by state & date
start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
if state_sel != "All":
    df_filtered = df_filtered[df_filtered['State'] == state_sel]

ev_demand_df = compute_ev_demand(df_filtered, bev_kwh_day, phev_kwh_day)
ev_demand_df = compute_chargers_and_grid(ev_demand_df, charger_kw, diversity, avg_hours_per_day)

infra_trend = ev_demand_df.groupby('Date').agg({
    'Monthly_GWh_Demand': 'sum',
    'Chargers_Required_int': 'sum',
    'Grid_Load_MW': 'sum'
}).reset_index().rename(columns={'Chargers_Required_int': 'Chargers_Required'})

county_demand = ev_demand_df.groupby('County').agg({
    'Monthly_GWh_Demand': 'sum',
    'Chargers_Required_int': 'sum',
    'Grid_Load_MW': 'sum'
}).reset_index().rename(columns={'Chargers_Required_int': 'Chargers_Required'})

# Key KPIs
total_gwh = infra_trend['Monthly_GWh_Demand'].sum()
total_chargers = int(county_demand['Chargers_Required'].sum()) if not county_demand.empty else 0
total_grid_mw = county_demand['Grid_Load_MW'].sum() if not county_demand.empty else 0
top_county = county_demand.sort_values('Monthly_GWh_Demand', ascending=False).head(1)['County'].iloc[0] if not county_demand.empty else "N/A"

# KPI cards
st.markdown("<div style='display:flex;gap:12px;margin-top:12px;'>", unsafe_allow_html=True)
st.markdown(f"<div class='kpi-card' style='flex:1'><div class='small-muted'>Total Energy (GWh)</div><div style='font-size:20px;font-weight:700;color:#1b4332'>{total_gwh:,.2f}</div></div>", unsafe_allow_html=True)
st.markdown(f"<div class='kpi-card' style='flex:1'><div class='small-muted'>Estimated Chargers</div><div style='font-size:20px;font-weight:700;color:#1b4332'>{total_chargers:,}</div></div>", unsafe_allow_html=True)
st.markdown(f"<div class='kpi-card' style='flex:1'><div class='small-muted'>Aggregate Grid Load (MW)</div><div style='font-size:20px;font-weight:700;color:#1b4332'>{total_grid_mw:,.2f}</div></div>", unsafe_allow_html=True)
st.markdown(f"<div class='kpi-card' style='flex:1'><div class='small-muted'>Top County by Demand</div><div style='font-size:20px;font-weight:700;color:#1b4332'>{top_county}</div></div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# --------------------------
# Main layout: National Trends + County Hotspots
# --------------------------
st.header("National Trends & Infrastructure Needs")
col1, col2 = st.columns([3, 1])
with col1:
    st.subheader("National EV Charging Energy Demand (GWh) — Over Time")
    fig1 = px.line(infra_trend, x='Date', y='Monthly_GWh_Demand', markers=True, labels={'Monthly_GWh_Demand': 'Monthly Demand (GWh)'})
    fig1.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    st.subheader("National KPIs (details)")
    st.write("Summary of aggregated metrics for selected date range / state.")
    st.write(f"Total GWh: **{total_gwh:,.2f}**")
    st.write(f"Estimated Chargers: **{total_chargers:,}**")
    st.write(f"Aggregate Grid Load (MW): **{total_grid_mw:,.2f}**")
    if st.button("Export national trend CSV"):
        csv_bytes = df_to_csv_bytes(infra_trend)
        st.download_button("Download national_trend.csv", data=csv_bytes, file_name="national_trend.csv", mime="text/csv")

st.markdown("---")

st.header("County Hotspots & Prioritization")
top_k = st.slider("Top N counties by total monthly demand", min_value=5, max_value=50, value=10)
county_demand_sorted = county_demand.sort_values('Monthly_GWh_Demand', ascending=False).head(top_k)
fig2 = px.bar(county_demand_sorted, x='County', y='Monthly_GWh_Demand', hover_data=['Chargers_Required','Grid_Load_MW'],
              labels={'Monthly_GWh_Demand': 'Total Monthly Demand (GWh)'})
st.plotly_chart(fig2, use_container_width=True)
st.dataframe(county_demand_sorted.reset_index(drop=True).rename(columns={'Monthly_GWh_Demand':'Total_GWh','Chargers_Required':'Chargers','Grid_Load_MW':'GridLoad_MW'}))

# --------------------------
# County Forecast Viewer
# --------------------------
st.markdown("---")
st.header("County Forecast Viewer (3-year monthly forecast)")
county_list = sorted(df['County'].dropna().unique().tolist())
county_sel = st.selectbox("Select County for forecast", county_list, index=0)

if county_sel:
    county_df = df[df['County'] == county_sel].sort_values("Date")
    if county_df.empty:
        st.warning("No data for selected county.")
    else:
        if model is None:
            st.warning("Forecast model not loaded. Forecast disabled.")
        else:
            # Forecast logic (your existing approach)
            historical_ev = list(county_df['Electric Vehicle (EV) Total'].values[-6:])
            cumulative_ev = list(np.cumsum(historical_ev))
            months_since_start = county_df['months_since_start'].max()
            latest_date = county_df['Date'].max()

            future_rows = []
            forecast_horizon = 36

            for i in range(1, forecast_horizon + 1):
                forecast_date = latest_date + pd.DateOffset(months=i)
                months_since_start += 1
                lag1, lag2, lag3 = historical_ev[-1], historical_ev[-2], historical_ev[-3]
                roll_mean = np.mean([lag1, lag2, lag3])
                pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
                pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
                recent_cumulative = cumulative_ev[-6:]
                ev_growth_slope = np.polyfit(range(len(recent_cumulative)), recent_cumulative, 1)[0] if len(recent_cumulative) == 6 else 0

                new_row = {
                    'months_since_start': months_since_start,
                    'county_encoded': county_df['county_encoded'].iloc[0],
                    'ev_total_lag1': lag1,
                    'ev_total_lag2': lag2,
                    'ev_total_lag3': lag3,
                    'ev_total_roll_mean_3': roll_mean,
                    'ev_total_pct_change_1': pct_change_1,
                    'ev_total_pct_change_3': pct_change_3,
                    'ev_growth_slope': ev_growth_slope
                }

                pred = model.predict(pd.DataFrame([new_row]))[0]
                future_rows.append({"Date": forecast_date, "Predicted EV Total": round(pred)})

                historical_ev.append(pred)
                if len(historical_ev) > 6:
                    historical_ev.pop(0)
                cumulative_ev.append(cumulative_ev[-1] + pred)
                if len(cumulative_ev) > 6:
                    cumulative_ev.pop(0)

            historical_cum = county_df[['Date', 'Electric Vehicle (EV) Total']].copy()
            historical_cum['Source'] = 'Historical'
            historical_cum['Cumulative EV'] = historical_cum['Electric Vehicle (EV) Total'].cumsum()

            forecast_df = pd.DataFrame(future_rows)
            forecast_df['Source'] = 'Forecast'
            forecast_df['Cumulative EV'] = forecast_df['Predicted EV Total'].cumsum() + historical_cum['Cumulative EV'].iloc[-1]

            combined = pd.concat([
                historical_cum[['Date', 'Cumulative EV', 'Source']],
                forecast_df[['Date', 'Cumulative EV', 'Source']]
            ], ignore_index=True)

            # Plot cumulative
            fig, ax = plt.subplots(figsize=(10,4))
            for label, data in combined.groupby('Source'):
                ax.plot(data['Date'], data['Cumulative EV'], label=label, marker='o' if label=='Forecast' else '.')
            ax.set_title(f"Cumulative EV Trend - {county_sel} (3-Year Forecast)")
            ax.set_xlabel("Date"); ax.set_ylabel("Cumulative EV Count")
            ax.grid(True, alpha=0.3)
            ax.legend()
            st.pyplot(fig)

            # Show forecast table and allow CSV download
            st.subheader("Forecast sample (next 12 months)")
            st.dataframe(forecast_df.head(12).reset_index(drop=True))
            csv_bytes = df_to_csv_bytes(forecast_df)
            st.download_button("Download forecast CSV", data=csv_bytes, file_name=f"{county_sel}_forecast.csv", mime="text/csv")

            # Option: Generate AI explanation for this forecast
            if st.button("Explain forecast (AI)"):
                prompt = (f"County: {county_sel}. Provide a concise explanation (<=150 words) of the forecast trend "
                          f"based on recent historical counts and the 3-year monthly forecast sample: {forecast_df.head(6).to_dict(orient='records')}. "
                          "Mention drivers and recommended planner actions.")
                ai_text = call_genai_chat(prompt, model=llm_model, max_tokens=llm_max_tokens, temperature=llm_temp)
                st.session_state['last_ai'] = ai_text
                st.info(ai_text)

# --------------------------
# Charger Type Mix Analysis
# --------------------------
st.markdown("---")
st.header("Charger Type Mix & Grid Impact")
st.write("Split demand between DC Fast Chargers (DCFC) and Level 2 chargers; compare grid impacts over time.")

dcfc_share = st.slider("DC Fast Charger share of demand", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
level2_share = 1.0 - dcfc_share
DCFC_KW = st.number_input("DC Fast Charger kW", value=50)
LEVEL2_KW = st.number_input("Level 2 Charger kW", value=7)

mix_df = infra_trend.copy()
mix_df['DCFC_GWh'] = mix_df['Monthly_GWh_Demand'] * dcfc_share
mix_df['Level2_GWh'] = mix_df['Monthly_GWh_Demand'] * level2_share

mix_df['DCFC_Chargers'] = mix_df['DCFC_GWh'] * 1e6 / (DCFC_KW * 24 * 30)
mix_df['Level2_Chargers'] = mix_df['Level2_GWh'] * 1e6 / (LEVEL2_KW * 24 * 30)

mix_df['DCFC_Grid_MW'] = mix_df['DCFC_Chargers'] * DCFC_KW * diversity / 1000
mix_df['Level2_Grid_MW'] = mix_df['Level2_Chargers'] * LEVEL2_KW * diversity / 1000

fig = go.Figure()
fig.add_trace(go.Scatter(x=mix_df['Date'], y=mix_df['DCFC_Chargers'], mode='lines+markers', name='DCFC Chargers'))
fig.add_trace(go.Scatter(x=mix_df['Date'], y=mix_df['Level2_Chargers'], mode='lines+markers', name='Level2 Chargers'))
fig.update_layout(title="Chargers Required by Type Over Time", xaxis_title="Date", yaxis_title="Number of Chargers")
st.plotly_chart(fig, use_container_width=True)

figg = go.Figure()
figg.add_trace(go.Scatter(x=mix_df['Date'], y=mix_df['DCFC_Grid_MW'], mode='lines+markers', name='DCFC Grid MW'))
figg.add_trace(go.Scatter(x=mix_df['Date'], y=mix_df['Level2_Grid_MW'], mode='lines+markers', name='Level2 Grid MW'))
figg.update_layout(title="Grid Load by Charger Type Over Time", xaxis_title="Date", yaxis_title="Grid Load (MW)")
st.plotly_chart(figg, use_container_width=True)

# --------------------------
# Scenario Analysis (what-if) + AI narration
# --------------------------
st.markdown("---")
st.header("Scenario Analysis & Simulation")
st.write("Quick scenario sets and AI narration for planner-friendly explanations.")

scenario_options = [
    {"name":"Base", "DCFC_KW":DCFC_KW, "LEVEL2_KW":LEVEL2_KW, "DIVERSITY":diversity},
    {"name":"High Power DCFC", "DCFC_KW":100, "LEVEL2_KW":LEVEL2_KW, "DIVERSITY":diversity},
    {"name":"Low Diversity", "DCFC_KW":DCFC_KW, "LEVEL2_KW":LEVEL2_KW, "DIVERSITY":0.3},
    {"name":"High Diversity", "DCFC_KW":DCFC_KW, "LEVEL2_KW":LEVEL2_KW, "DIVERSITY":0.7},
]

scenario_results = []
for s in scenario_options:
    tmp = mix_df.copy()
    tmp['Total_Grid_MW'] = tmp['DCFC_Chargers'] * s['DCFC_KW'] * s['DIVERSITY'] / 1000 + tmp['Level2_Chargers'] * s['LEVEL2_KW'] * s['DIVERSITY'] / 1000
    total_peak = tmp['Total_Grid_MW'].max()
    scenario_results.append({"Scenario": s['name'], "Peak_Grid_MW": float(total_peak), "Avg_Grid_MW": float(tmp['Total_Grid_MW'].mean())})

scenario_df = pd.DataFrame(scenario_results)
st.table(scenario_df)

st.subheader("AI Narration of Scenarios")
user_scenario_text = st.text_input("Describe a scenario or ask a question:", value="Describe Base vs High Power DCFC impacts and recommendations.")
if st.button("Generate AI Narrative (Scenarios)"):
    prompt = f"Scenarios summary: {scenario_df.to_dict(orient='records')}. User question: {user_scenario_text}. Provide a concise, actionable explanation (<=150 words) for planners."
    llm_resp = call_genai_chat(prompt, model=llm_model, max_tokens=llm_max_tokens, temperature=llm_temp)
    st.session_state['last_ai'] = llm_resp
    st.info(llm_resp)

# --------------------------
# AI Assistant & Executive Summary
# --------------------------
st.markdown("---")
st.header("AI Assistant (Q&A & Auto Summary)")
st.write("Ask plain-English questions about the dataset or generate an executive summary for decision-makers.")

qa_input = st.text_input("Ask a question to the AI Assistant:")
if st.button("Ask AI"):
    top_snapshot = county_demand.sort_values('Monthly_GWh_Demand', ascending=False).head(10).to_dict(orient='records')
    latest_infra = infra_trend.tail(6).to_dict(orient='records')
    prompt = f"Top counties: {top_snapshot}. Recent national infra trend (last 6): {latest_infra}. Question: {qa_input}. Answer concisely with numbers when possible."
    answer = call_genai_chat(prompt, model=llm_model, max_tokens=llm_max_tokens, temperature=llm_temp)
    st.session_state['last_ai'] = answer
    st.success(answer)

st.subheader("Auto-generated Executive Summary")
if st.button("Generate Executive Summary"):
    top10 = county_demand.sort_values('Monthly_GWh_Demand', ascending=False).head(10)
    prompt = f"Create a 5-7 sentence executive summary for planners given these top 10 counties: {top10.to_dict(orient='records')}. Mention actions and grid concerns."
    summary = call_genai_chat(prompt, model=llm_model, max_tokens=llm_max_tokens, temperature=llm_temp)
    st.session_state['last_ai'] = summary
    st.info(summary)

with st.expander("Latest AI output (click to view)"):
    if st.session_state['last_ai']:
        st.markdown(st.session_state['last_ai'])
        st.download_button("Download AI narrative (txt)", data=st.session_state['last_ai'], file_name="ai_narrative.txt")
    else:
        st.write("No AI output yet. Generate one using the buttons above.")

# --------------------------
# Footer
# --------------------------
st.markdown("---")
st.markdown("""
**Resources & Notes**
- Data sample: Washington State DOL (monthly county registrations).  
- GenAI: Groq models (select in sidebar). Make sure you have a valid GROQ_API_KEY.  
- Authors: Sanjai Guru & Nishanth  — NXTWAVE x OpenAI Buildathon.
""")

# Quick troubleshooting helper
if groq_client is None:
    st.warning("Groq client is not initialized. LLM features will show an explanatory message instead of real AI responses. "
               "Set GROQ_API_KEY environment variable with a valid key or hardcode for local testing (not recommended).")
