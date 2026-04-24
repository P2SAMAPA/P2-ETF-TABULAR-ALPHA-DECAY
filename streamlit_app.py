"""
Streamlit Dashboard for Tabular Alpha Decay Engine.
"""

import streamlit as st
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
import json
import config
from us_calendar import USMarketCalendar

st.set_page_config(page_title="P2Quant Tabular Alpha Decay", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 600; color: #1f77b4; }
    .hero-card { background: linear-gradient(135deg, #1f77b4 0%, #2C5282 100%); border-radius: 16px; padding: 2rem; color: white; text-align: center; }
    .hero-ticker { font-size: 4rem; font-weight: 800; }
    .metric-positive { color: #28a745; font-weight: 600; }
    .metric-negative { color: #dc3545; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_latest_results():
    try:
        api = HfApi(token=config.HF_TOKEN)
        files = api.list_repo_files(repo_id=config.HF_OUTPUT_REPO, repo_type="dataset")
        json_files = sorted([f for f in files if f.startswith("tabular_alpha_decay_") and f.endswith('.json')], reverse=True)
        if not json_files:
            return None
        local_path = hf_hub_download(
            repo_id=config.HF_OUTPUT_REPO, filename=json_files[0],
            repo_type="dataset", token=config.HF_TOKEN, cache_dir="./hf_cache"
        )
        with open(local_path) as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

def return_badge(val):
    if val >= 0:
        return f'<span class="metric-positive">+{val*100:.2f}%</span>'
    return f'<span class="metric-negative">{val*100:.2f}%</span>'

# --- Sidebar ---
st.sidebar.markdown("## ⚙️ Configuration")
calendar = USMarketCalendar()
st.sidebar.markdown(f"**📅 Next Trading Day:** {calendar.next_trading_day().strftime('%Y-%m-%d')}")
data = load_latest_results()
if data:
    st.sidebar.markdown(f"**Run Date:** {data.get('run_date', 'Unknown')}")
st.sidebar.divider()
st.sidebar.markdown("### 📊 Model Parameters")
st.sidebar.markdown(f"- Objective: **{config.LGB_PARAMS['objective']}**")
st.sidebar.markdown(f"- Decay Max Lag: **{config.DECAY_MAX_LAG} days**")

st.markdown('<div class="main-header">📊 P2Quant Tabular Alpha Decay</div>', unsafe_allow_html=True)
st.markdown('<div>LightGBM Cross‑Sectional Ranking + Alpha Half‑Life Decay</div>', unsafe_allow_html=True)

with st.expander("📘 How It Works", expanded=False):
    st.markdown("""
    **Tabular Alpha**: LightGBM ranker trained on lagged returns and macro features to predict next‑day return.
    **Alpha Decay**: The half‑life of the signal's predictive power is estimated from the auto‑correlation decay across lags.
    **Decay‑Adjusted Return** = Raw Predicted Return × exp(-ln(2)/half_life)
    """)

if data is None:
    st.warning("No data available.")
    st.stop()

daily = data['daily_trading']
universes = daily['universes']
top_picks = daily['top_picks']

tabs = st.tabs(["📊 Combined", "📈 Equity Sectors", "💰 FI/Commodities"])
universe_keys = ["COMBINED", "EQUITY_SECTORS", "FI_COMMODITIES"]

for tab, key in zip(tabs, universe_keys):
    with tab:
        top = top_picks.get(key, [])
        universe_data = universes.get(key, {})
        if top:
            pick = top[0]
            ticker = pick['ticker']
            raw = pick['raw_pred']
            adj = pick['decay_adjusted']
            hl = pick['half_life']
            st.markdown(f"""
            <div class="hero-card">
                <div style="font-size: 1.2rem; opacity: 0.8;">📊 TOP PICK (Decay‑Adjusted)</div>
                <div class="hero-ticker">{ticker}</div>
                <div>Raw Forecast: {return_badge(raw)}</div>
                <div>Decay‑Adjusted: {return_badge(adj)}</div>
                <div style="margin-top: 0.5rem;">Signal Half‑Life: {hl:.1f} days</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### Top 3 Picks")
            rows = []
            for p in top:
                rows.append({
                    "Ticker": p['ticker'],
                    "Raw Forecast": f"{p['raw_pred']*100:.2f}%",
                    "Decay‑Adjusted": f"{p['decay_adjusted']*100:.2f}%",
                    "Half‑Life (days)": f"{p['half_life']:.1f}"
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.markdown("### All ETFs")
            all_rows = []
            for t, d in universe_data.items():
                all_rows.append({
                    "Ticker": t,
                    "Raw Forecast": f"{d['raw_pred']*100:.2f}%",
                    "Decay‑Adjusted": f"{d['decay_adjusted']*100:.2f}%",
                    "Half‑Life": f"{d['half_life']:.1f}"
                })
            df_all = pd.DataFrame(all_rows).sort_values("Decay‑Adjusted", ascending=False)
            st.dataframe(df_all, use_container_width=True, hide_index=True)
        else:
            st.info(f"No data for {key}.")
