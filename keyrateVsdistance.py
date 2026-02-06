
Sailaja -572
03:52 (2 hours ago)
to me

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(page_title="QKD Simulator", layout="wide")

st.title("🔐 Key Rate VS Distance Simulator")
st.caption("Interactive visualization of QKD performance, security limits, and system feasibility")

# -------------------------------------------------
# Fibre data
# -------------------------------------------------
fibres = {
    "SMF": 0.20,
    "Ultra-low-loss": 0.16,
    "PMF": 0.30,
    "Hollow-core": 0.12
}

# -------------------------------------------------
# Sidebar parameters
# -------------------------------------------------
st.sidebar.header("Simulation Parameters")

fibre = st.sidebar.selectbox("Fibre Type", list(fibres.keys()))
alpha_base = fibres[fibre]

wavelength = st.sidebar.slider("Wavelength (nm)", 1300, 1600, 1550)
mu = st.sidebar.slider("Mean photon number (μ)", 0.01, 1.0, 0.5)
eta = st.sidebar.slider("Detector efficiency (η)", 0.1, 1.0, 0.6)
dark = st.sidebar.slider("Dark count rate (P_dark)", 1e-6, 1e-3, 1e-4, format="%.6f")
e_opt = st.sidebar.slider("Optical error (e_opt)", 0.0, 0.05, 0.02)
max_d = st.sidebar.slider("Maximum distance (km)", 50, 300, 150)

QBER_THRESHOLD = 0.11

# -------------------------------------------------
# Physics model
# -------------------------------------------------
alpha = alpha_base + 0.00002 * (wavelength - 1550) ** 2
d = np.linspace(0, max_d, 80)

T = 10 ** (-alpha * d / 10)
signal = mu * T * eta
noise = dark

qber = (0.5 * noise + e_opt * signal) / (signal + noise)
key_rate = signal * (1 - qber)

photons_survived = mu * T
photons_lost = mu - photons_survived

# -------------------------------------------------
# Security analysis
# -------------------------------------------------
secure_mask = qber <= QBER_THRESHOLD
max_secure_distance = d[secure_mask][-1] if np.any(secure_mask) else 0.0
i = np.where(d == max_secure_distance)[0][0] if max_secure_distance > 0 else -1

# -------------------------------------------------
# Key metrics for table
# -------------------------------------------------
raw_key_rate = signal[i]
sifted_key_rate = 0.5 * raw_key_rate
secure_key_rate = sifted_key_rate * (1 - qber[i])

# =================================================
# PLOT 1: Key Rate vs Distance + Toolbar Table Toggle
# =================================================
fig1 = go.Figure()

# Curve
fig1.add_trace(go.Scatter(
    x=d,
    y=key_rate,
    mode="lines",
    name="Key Rate",
    line=dict(color="cyan", width=2),
    visible=True
))

# Table
fig1.add_trace(go.Table(
    header=dict(
        values=["Parameter", "Value"],
        fill_color="black",
        font=dict(color="white"),
        align="left"
    ),
    cells=dict(
        values=[
            [
                "Fibre Type",
                "Wavelength (nm)",
                "Effective Attenuation (dB/km)",
                "Maximum Secure Distance (km)",
                "Mean Photon Number (μ)",
                "Detector Efficiency (η)",
                "Dark Count Rate",
                "Raw Key Rate",
                "Sifted Key Rate",
                "QBER",
                "QBER Threshold",
                "Final Secure Key Rate"
            ],
            [
                fibre,
                wavelength,
                f"{alpha:.3f}",
                f"{max_secure_distance:.1f}",
                mu,
                eta,
                f"{dark:.1e}",
                f"{raw_key_rate:.3e}",
                f"{sifted_key_rate:.3e}",
                f"{qber[i]:.4f}",
                QBER_THRESHOLD,
                f"{secure_key_rate:.3e}"
            ]
        ],
        fill_color="rgba(30,30,30,0.9)",
        font=dict(color="white"),
        align="left"
    ),
    visible=False
))

# Secure / insecure regions
fig1.add_vrect(0, max_secure_distance, fillcolor="green", opacity=0.15, layer="below")
fig1.add_vrect(max_secure_distance, max_d, fillcolor="red", opacity=0.15, layer="below")
fig1.add_vline(x=max_secure_distance, line_dash="dash", line_color="yellow")

# Toolbar buttons (SIDE BY SIDE)
fig1.update_layout(
    title="Key Rate vs Distance (System Feasibility)",
    template="plotly_dark",
    xaxis_title="Distance (km)",
    yaxis_title="Key Rate",
    updatemenus=[
        dict(
            type="buttons",
            direction="right",   # 👈 SIDE BY SIDE
            x=0.55,
            y=1.25,
            showactive=True,
            buttons=[
                dict(
                    label="📈 Curve",
                    method="update",
                    args=[{"visible": [True, False]}]
                ),
                dict(
                    label="📋 Table",
                    method="update",
                    args=[{"visible": [False, True]}]
                )
            ]
        )
    ]
)

st.plotly_chart(fig1, use_container_width=True)

# =================================================
# PLOT 2: QBER vs Distance
# =================================================
fig2 = px.area(
    x=d,
    y=qber,
    labels={"x": "Distance (km)", "y": "QBER"},
    title="QBER vs Distance",
    template="plotly_dark"
)
fig2.add_hline(y=QBER_THRESHOLD, line_dash="dash", line_color="red")
st.plotly_chart(fig2, use_container_width=True)

# =================================================
# PLOT 3: Attenuation vs Wavelength
# =================================================
wl = np.linspace(1300, 1600, 80)
rows = []

for f_name, base_loss in fibres.items():
    for w in wl:
        rows.append({
            "Wavelength (nm)": w,
            "Loss (dB/km)": base_loss + 0.00002 * (w - 1550) ** 2,
            "Fibre Type": f_name
        })

df_wl = pd.DataFrame(rows)

fig3 = px.line(
    df_wl,
    x="Wavelength (nm)",
    y="Loss (dB/km)",
    color="Fibre Type",
    title="Attenuation vs Wavelength for Different Fibres",
    template="plotly_dark"
)
st.plotly_chart(fig3, use_container_width=True)

# =================================================
# PLOT 4: Photon Survival
# =================================================
df_bar = pd.DataFrame({
    "Type": ["Survived", "Lost"],
    "Photons": [photons_survived[-1], photons_lost[-1]]
})

fig4 = px.bar(
    df_bar,
    x="Type",
    y="Photons",
    title="Photons at Maximum Distance",
    template="plotly_dark",
    color="Type",
    text_auto=True
)
st.plotly_chart(fig4, use_container_width=True)

# =================================================
# PLOT 5: Key Rate Factors
# =================================================
pie_data = pd.DataFrame({
    "Factor": [
        "Useful Signal",
        "Channel Loss",
        "Detector Inefficiency",
        "Noise (Dark counts)"
    ],
    "Contribution": [
        float(np.mean(signal)),
        float(np.mean(mu - photons_survived)),
        float(np.mean(mu * (1 - eta))),
        float(np.mean(noise))
    ]
})

fig5 = px.pie(
    pie_data,
    names="Factor",
    values="Contribution",
    title="Factors Affecting Key Rate",
    template="plotly_dark"
)
st.plotly_chart(fig5, use_container_width=True)

# -------------------------------------------------
# Summary
# -------------------------------------------------
st.success(f"""
**Summary**
• Fibre: {fibre}  
• Wavelength: {wavelength} nm  
• Maximum secure transmission distance: {max_secure_distance:.1f} km  

All original graphs are preserved.  
Only the first graph includes a **horizontal toolbar toggle** for Curve ↔ Table.
""")
