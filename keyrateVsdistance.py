import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from math import log2, exp, factorial

# =========================================================
# Helper functions
# =========================================================
def binary_entropy(q):
    if q <= 0 or q >= 1:
        return 0
    return -q * log2(q) - (1 - q) * log2(1 - q)

def poisson(n, mu):
    return (mu ** n * exp(-mu)) / factorial(n)

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Quantum Networking Simulator", layout="wide")
st.title("🌐 Quantum Networking Simulator")
st.caption("Quantum Repeaters + PNS Attack & Decoy-State Defense")

# =========================================================
# TABS
# =========================================================
tab1, tab2 = st.tabs([
    "🔗 Quantum Repeaters",
    "🛡️ PNS Attack & Decoy States"
])

# =========================================================
# ================= TAB 1: QUANTUM REPEATERS ==============
# =========================================================
with tab1:
    st.subheader("🔗 Quantum Repeater Performance Dashboard")
    st.caption("Entanglement distribution using Bell-state measurements")

    # ---------------- Sidebar ----------------
    st.sidebar.header("Repeater Parameters")

    alpha = st.sidebar.slider("Fiber loss α (dB/km)", 0.15, 0.30, 0.2)
    F0 = st.sidebar.slider("Initial Bell Fidelity (F₀)", 0.8, 1.0, 0.95)
    p_dep = st.sidebar.slider("Depolarization probability", 0.0, 0.2, 0.05)
    gamma = st.sidebar.slider("Memory decoherence rate γ", 0.0, 1.0, 0.1)
    p_BSM = st.sidebar.slider("BSM success probability", 0.1, 0.5, 0.5)
    eta_d = st.sidebar.slider("Detector efficiency η_d", 0.1, 1.0, 0.9)
    R_pair = st.sidebar.slider("Pair generation rate (pairs/sec)", 1e4, 1e7, 1e6)

    # Fixed configuration
    segment_distance = 50
    num_repeaters = 3
    num_segments = num_repeaters + 1
    T_wait = 200  # ms

    # ---------------- Core calculations ----------------
    eta_channel = 10 ** (-alpha * segment_distance / 10)
    F_eff = F0 * (1 - p_dep)

    T_mem = 100  # ms
    P_mem = 1 - exp(-T_mem / T_wait)

    F_final = (F_eff * exp(-gamma * (T_mem / 1000))) ** num_segments
    QBER = (1 - F_final) / 2
    key_fraction = max(1e-12, 1 - 2 * binary_entropy(QBER))

    key_rate = (
        R_pair
        * (eta_channel ** num_segments)
        * (p_BSM ** num_repeaters)
        * (P_mem ** num_repeaters)
        * eta_d
        * key_fraction
    )

    # ---------------- Metrics ----------------
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Repeaters", num_repeaters)
    m2.metric("Final Fidelity", f"{F_final:.4f}")
    m3.metric("QBER", f"{QBER:.4f}")
    m4.metric("Key Rate (bps)", f"{key_rate:.2e}")

    st.divider()

    # ---------------- Key Rate vs Distance ----------------
    st.subheader("🔹 Key Rate vs Distance (Log Scale)")

    distances = np.arange(50, 550, 50)
    rates = []

    for d in distances:
        segs = max(1, int(d / segment_distance))
        Fd = F_eff ** segs
        Qd = (1 - Fd) / 2
        rate = (
            R_pair
            * (eta_channel ** segs)
            * (p_BSM ** (segs - 1))
            * eta_d
            * max(1e-15, 1 - 2 * binary_entropy(Qd))
        )
        rates.append(rate)

    df_dist = pd.DataFrame({
        "Distance (km)": distances,
        "Key Rate (bps)": rates
    })

    chart_dist = alt.Chart(df_dist).mark_line(point=True).encode(
        x="Distance (km):Q",
        y=alt.Y("Key Rate (bps):Q", scale=alt.Scale(type="log")),
        tooltip=["Distance (km)", "Key Rate (bps)"]
    )

    st.altair_chart(chart_dist, use_container_width=True)
    st.dataframe(df_dist, use_container_width=True)

    st.divider()

    # ---------------- QBER vs Repeaters ----------------
    st.subheader("🔹 QBER vs Number of Repeaters")

    reps = np.arange(0, 8)
    qbers = [(1 - (F_eff ** (r + 1))) / 2 for r in reps]

    df_qber = pd.DataFrame({
        "Repeaters": reps,
        "QBER": qbers
    })

    chart_qber = alt.Chart(df_qber).mark_bar().encode(
        x="Repeaters:O",
        y="QBER:Q",
        tooltip=["Repeaters", "QBER"]
    )

    st.altair_chart(chart_qber, use_container_width=True)
    st.dataframe(df_qber, use_container_width=True)

    st.divider()

    # ---------------- Key Rate vs Memory ----------------
    st.subheader("🔹 Key Rate vs Quantum Memory Lifetime")

    T_vals = np.linspace(10, 1000, 25)
    rates_mem = []

    for T in T_vals:
        Pm = 1 - exp(-T / T_wait)
        FT = (F_eff * exp(-gamma * (T / 1000))) ** num_segments
        QT = (1 - FT) / 2

        rate = (
            R_pair
            * (eta_channel ** num_segments)
            * (p_BSM ** num_repeaters)
            * (Pm ** num_repeaters)
            * eta_d
            * max(1e-15, 1 - 2 * binary_entropy(QT))
        )
        rates_mem.append(rate)

    df_mem = pd.DataFrame({
        "Memory Lifetime (ms)": T_vals.astype(int),
        "Key Rate (bps)": rates_mem
    })

    chart_mem = alt.Chart(df_mem).mark_line(point=True).encode(
        x="Memory Lifetime (ms):Q",
        y=alt.Y("Key Rate (bps):Q", scale=alt.Scale(type="log")),
        tooltip=["Memory Lifetime (ms)", "Key Rate (bps)"]
    )

    st.altair_chart(chart_mem, use_container_width=True)
    st.dataframe(df_mem, use_container_width=True)

# =========================================================
# ================= TAB 2: PNS + DECOY STATES ==============
# =========================================================
with tab2:
    st.subheader("🛡️ PNS Attack & Decoy-State Defense")
    st.caption("Understand the attack first, then see how decoy states defeat it")

    st.sidebar.header("PNS & Decoy Parameters")

    mu_s = st.sidebar.slider("Signal intensity μ", 0.1, 1.0, 0.5)
    mu_d = st.sidebar.slider("Decoy intensity ν", 0.01, 0.4, 0.1)
    distance = st.sidebar.slider("Distance (km)", 10, 300, 100)
    alpha = st.sidebar.slider("Fiber loss (dB/km)", 0.15, 0.30, 0.2)
    eta_det = st.sidebar.slider("Detector efficiency", 0.1, 1.0, 0.6)
    pns_on = st.sidebar.checkbox("Enable PNS Attack", value=True)

    eta_ch = 10 ** (-alpha * distance / 10)

    # Photon statistics
    P0s, P1s = poisson(0, mu_s), poisson(1, mu_s)
    Pms = 1 - P0s - P1s

    P0d, P1d = poisson(0, mu_d), poisson(1, mu_d)
    Pmd = 1 - P0d - P1d

    # ---------------- Step 1: PNS Attack ----------------
    st.markdown("### 🔴 Step 1: Why PNS Attacks Work")

    col1, col2 = st.columns(2)

    with col1:
        df_signal = pd.DataFrame({
            "Photon Type": ["Vacuum", "Single Photon", "Multi Photon"],
            "Probability": [P0s, P1s, Pms]
        })

        chart_pns = alt.Chart(df_signal).mark_bar().encode(
            x="Photon Type",
            y="Probability",
            tooltip=["Photon Type", "Probability"]
        )

        st.altair_chart(chart_pns, use_container_width=True)

    with col2:
        st.info("""
• Weak laser pulses sometimes emit **multiple photons**  
• Eve keeps one photon and forwards the rest  
• Alice and Bob cannot detect this directly  
""")

    st.table(pd.DataFrame({
        "Pulse Type": ["Vacuum", "Single Photon", "Multi Photon"],
        "Eve Can Steal?": ["No", "No", "Yes"],
        "Bob Notices?": ["No", "No", "No"]
    }))

    # ---------------- Step 2: Decoy States ----------------
    st.markdown("### 🟢 Step 2: How Decoy States Detect Eve")

    col3, col4 = st.columns(2)

    with col3:
        df_compare = pd.DataFrame({
            "Photon Type": ["Vacuum", "Single Photon", "Multi Photon"],
            "Signal": [P0s, P1s, Pms],
            "Decoy": [P0d, P1d, Pmd]
        }).melt("Photon Type", var_name="Pulse", value_name="Probability")

        chart_decoy = alt.Chart(df_compare).mark_bar().encode(
            x="Photon Type",
            y="Probability",
            color="Pulse",
            tooltip=["Pulse", "Probability"]
        )

        st.altair_chart(chart_decoy, use_container_width=True)

    with col4:
        st.info("""
• Eve cannot distinguish signal from decoy pulses  
• Any selective attack breaks statistics  
• Alice and Bob detect Eve **without revealing the key**  
""")

    # Gain comparison
    Y1 = eta_ch * eta_det * (0.2 if pns_on else 1)
    Ymulti = eta_ch * eta_det

    Qs = P1s * Y1 + Pms * Ymulti
    Qd = P1d * Y1 + Pmd * Ymulti

    st.markdown("### 📊 Gain Consistency Test")

    chart_gain = alt.Chart(pd.DataFrame({
        "Pulse": ["Signal", "Decoy"],
        "Gain": [Qs, Qd]
    })).mark_bar().encode(
        x="Pulse",
        y="Gain",
        tooltip=["Pulse", "Gain"]
    )

    st.altair_chart(chart_gain, use_container_width=True)

    if abs(Qs - Qd) > 0.01 and pns_on:
        st.error("🚨 PNS ATTACK DETECTED — Statistics inconsistent")
    else:
        st.success("✅ CHANNEL SECURE — No PNS signature detected")

    st.success("""
**Final takeaway**

• PNS attacks exploit multi-photon pulses  
• Decoy states turn statistics into a security alarm  
• This is why modern QKD systems always use decoy states  
""")
