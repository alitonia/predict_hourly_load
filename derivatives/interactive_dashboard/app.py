import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

#  TODO: finish this and add real api ( may route from fast api path )
# --- PAGE CONFIG ---
st.set_page_config(page_title="PJM Grid Demand Forecast", layout="wide")


# --- FAKE DATA GENERATOR (Replace with your real model loading) ---
def get_forecast_data(days=7, temp_adjust=0):
    # Simulating data for demonstration
    dates = pd.date_range(start=pd.Timestamp.now().ceil('H'), periods=24 * days, freq='H')
    base_load = np.sin(np.linspace(0, days * np.pi * 2, 24 * days)) * 10 + 40  # Simple sine wave

    # Apply temperature adjustment (scenario logic)
    # Assume 1 degree adds ~0.5GW of load
    scenario_load = base_load + (temp_adjust * 0.5)

    df = pd.DataFrame({'timestamp': dates, 'p50': scenario_load})
    # Create uncertainty bands (wider during peaks for realism)
    df['p90'] = df['p50'] + (df['p50'] * 0.10)
    df['p10'] = df['p50'] - (df['p50'] * 0.10)
    return df


# --- SIDEBAR CONTROLS ---
st.sidebar.header("Planning Scenarios")
forecast_days = st.sidebar.slider("Forecast Horizon (Days)", 1, 14, 7)
temp_adj = st.sidebar.slider("Extreme Weather Adjustment (°F)", min_value=0, max_value=15, value=0)
st.sidebar.markdown("---")
st.sidebar.write("*Use adjustment to simulate heatwave impacts on grid load.*")

# --- MAIN APP ---
st.title("⚡ PJM Regional Demand Forecast")

# Fetch data based on inputs
df = get_forecast_data(days=forecast_days, temp_adjust=temp_adj)

# 1. METRICS ROW
col1, col2, col3 = st.columns(3)
current_peak = df['p50'].max()
capacity_limit = 52.0  # Hardcoded example capacity
risk_level = "LOW" if current_peak < 48 else "HIGH"

col1.metric("Forecasted Peak (P50)", f"{current_peak:.2f} GW",
            delta=f"{temp_adj * 0.5:.1f} GW (due to scenario)" if temp_adj > 0 else None, delta_color="inverse")
col2.metric("Grid Capacity", f"{capacity_limit} GW")
col3.metric("Risk Status", risk_level)

# 2. INTERACTIVE PLOTLY CHART
fig = go.Figure()

# Add Lower Bound (P10) - Invisible line that sets the floor for shading
fig.add_trace(go.Scatter(
    x=df['timestamp'], y=df['p10'],
    mode='lines', line=dict(width=0),
    name='P10 (Low Risk)', showlegend=False
))

# Add Upper Bound (P90) - Fills down to the P10 trace
fig.add_trace(go.Scatter(
    x=df['timestamp'], y=df['p90'],
    mode='lines', line=dict(width=0),
    fill='tonexty', fillcolor='rgba(255, 0, 0, 0.2)',  # Red shading for risk
    name='Uncertainty Range (P10-P90)'
))

# Add Median Forecast (P50) - The main line
fig.add_trace(go.Scatter(
    x=df['timestamp'], y=df['p50'],
    mode='lines', line=dict(color='blue', width=3),
    name='Baseline Forecast (P50)'
))

# Add Capacity Line (Crucial for context)
fig.add_hline(y=capacity_limit, line_dash="dot", line_color="red", annotation_text="Max Capacity (52GW)")

fig.update_layout(
    title="Hourly Demand Forecast with Risk Envelope",
    yaxis_title="Load (GW)",
    xaxis_title="Time (EST)",
    height=500,
    hovermode="x unified"  # Shows all values when hovering over a specific hour
)

st.plotly_chart(fig, use_container_width=True)