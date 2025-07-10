import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.integrate import odeint

# Caching simulation results for performance
@st.cache_data
def run_ode_sim(model_type, initial_values, t, beta, gamma, sigma=None):
    """
    Run the ODE simulation for SIR or SEIR model.
    """
    if model_type == "SIR":
        result = odeint(sir_model, initial_values, t, args=(beta, gamma))
    else:  # SEIR
        result = odeint(seir_model, initial_values, t, args=(beta, gamma, sigma))
    return result

# Defining SIR model function
def sir_model(y, t, beta, gamma):
    """
    SIR model differential equations.
    S: Susceptible, I: Infected, R: Recovered
    """
    S, I, R = y
    N = S + I + R
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# Defining SEIR model function
def seir_model(y, t, beta, gamma, sigma):
    """
    SEIR model differential equations.
    S: Susceptible, E: Exposed, I: Infected, R: Recovered
    """
    S, E, I, R = y
    N = S + E + I + R
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]

# Calculating key metrics
def calculate_metrics(t, result, model_type):
    """
    Calculate peak infection time and R0.
    """
    if model_type == "SIR":
        _, I, _ = result.T
    else:  # SEIR
        _, _, I, _ = result.T
    peak_time = t[np.argmax(I)]
    total_reach = np.max(I)
    return peak_time, total_reach

# Setting up Streamlit UI
st.set_page_config(page_title="Viral Topic Spread Simulator", layout="wide", page_icon="📈")
st.title("Viral Topic Spread on Social Media using SIR/SEIR Model")
st.markdown("""
Adjust the sliders to tweak the virality and engagement drop-off of a trending topic.
Choose between SIR and SEIR models to explore different spread dynamics!
""")

# Adding model explanation
with st.expander("Learn About the SIR/SEIR Models"):
    st.markdown("""
    **SIR Model**: Splits the population into three groups:
    - **Susceptible (S)**: Users who haven't encountered the topic.
    - **Infected (I)**: Users actively engaging with or sharing the topic.
    - **Recovered (R)**: Users who have lost interest.

    **SEIR Model**: Includes an **Exposed (E)** group for users who have seen the topic but haven't engaged yet.
    It accounts for a delay before engagement, controlled by the exposure rate (σ).
    """)

# Defining preset scenarios
presets = {
    "Global Trend": {"beta": 0.6, "gamma": 0.04, "sigma": 0.25},
    "Niche Topic": {"beta": 0.25, "gamma": 0.12, "sigma": 0.15},
    "Slow-Building Buzz": {"beta": 0.35, "gamma": 0.03, "sigma": 0.08}
}

# Creating UI for model selection and parameters
col1, col2 = st.columns(2)
with col1:
    model_type = st.selectbox("Select Model", ["SIR", "SEIR"], help="Choose SIR for simple spread or SEIR for delayed engagement.")
    scenario = st.selectbox("Choose a Scenario", ["Custom"] + list(presets.keys()), help="Pick a preset or customize your own.")
with col2:
    N = st.slider("Audience Size", 100, 10000, 1000, 100, help="Total potential audience for the topic.")
    log_scale = st.checkbox("Logarithmic Y-Axis", value=False, help="Use logarithmic scale for better visualization of trends.")

# Initializing parameters
I0 = 5
R0 = 0
if model_type == "SIR":
    E0 = 0
    S0 = N - I0 - R0
    initial_values = [S0, I0, R0]
    sigma = None
else:  # SEIR
    E0 = 0
    S0 = N - I0 - R0 - E0
    initial_values = [S0, E0, I0, R0]

# Setting parameters based on scenario
if scenario != "Custom":
    beta = presets[scenario]["beta"]
    gamma = presets[scenario]["gamma"]
    sigma = presets[scenario]["sigma"] if model_type == "SEIR" else None
else:
    col3, col4 = st.columns(2)
    with col3:
        beta = st.slider("Virality Rate (β)", 0.0, 1.0, 0.3, 0.01, help="How likely users are to share the topic.")
    with col4:
        gamma = st.slider("Engagement Drop-Off Rate (γ)", 0.0, 1.0, 0.1, 0.01, help="How quickly users lose interest.")
        if model_type == "SEIR":
            sigma = st.slider("Exposure Rate (σ)", 0.0, 1.0, 0.2, 0.01, help="Rate at which exposed users start engaging.")

# Validating parameters
if beta <= 0 or gamma <= 0 or (model_type == "SEIR" and sigma <= 0):
    st.error("All rates (β, γ, σ) must be positive.")
else:
    # Running simulation
    t = np.linspace(0, 60, 200)
    result = run_ode_sim(model_type, initial_values, t, beta, gamma, sigma)

    # Extracting results
    if model_type == "SIR":
        S, I, R = result.T
    else:  # SEIR
        S, E, I, R = result.T

    # Creating Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=S, mode='lines', name='Unaware (S)', line=dict(color='green')))
    if model_type == "SEIR":
        fig.add_trace(go.Scatter(x=t, y=E, mode='lines', name='Exposed (E)', line=dict(color='yellow')))
    fig.add_trace(go.Scatter(x=t, y=I, mode='lines', name='Engaging (I)', line=dict(color='magenta')))
    fig.add_trace(go.Scatter(x=t, y=R, mode='lines', name='Disengaged (R)', line=dict(color='gray')))
    fig.update_layout(
        title='SIR/SEIR Dynamics of a Trending Topic',
        xaxis_title='Time (days)',
        yaxis_title='Audience Size',
        yaxis_type='log' if log_scale else 'linear',
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Displaying metrics
    peak_time, total_reach = calculate_metrics(t, result, model_type)
    st.subheader("Simulation Insights")
    col5, col6, col7 = st.columns(3)
    with col5:
        st.metric("Peak Engagement Time", f"{peak_time:.1f} days")
    with col6:
        st.metric("Total Reach", f"{int(total_reach)} users")
    with col7:
        st.metric("Basic Reproduction Number (R₀)", f"{beta/gamma:.2f}")
        with st.expander("What is R₀?"):
            st.markdown("""
            **Basic Reproduction Number (R₀)**: Indicates how many new users engage with a topic for each user actively sharing it, in a fully unaware audience. Calculated as β/γ:
            - **R₀ > 1**: The topic spreads rapidly (trending).
            - **R₀ < 1**: The topic fades quickly.
            R₀ predicts the topic's early-stage viral potential.
            """)

    # Providing downloadable results
    df = pd.DataFrame({
        "Time": t,
        "Unaware": S,
        "Engaging": I,
        "Disengaged": R
    })
    if model_type == "SEIR":
        df["Exposed"] = E
    st.download_button(
        label="Download Data",
        data=df.to_csv().encode('utf-8'),
        file_name="trending_topic_data.csv",
        mime="text/csv"
    )

    # Feature: Comparative Scenario Analysis
    st.subheader("Compare Scenarios")
    selected_scenarios = st.multiselect("Select Scenarios to Compare", list(presets.keys()), default=["Global Trend", "Niche Topic"])
    fig_compare = go.Figure()
    metrics_data = []
    for scenario in selected_scenarios:
        temp_beta = presets[scenario]["beta"]
        temp_gamma = presets[scenario]["gamma"]
        temp_sigma = presets[scenario]["sigma"] if model_type == "SEIR" else None
        result = run_ode_sim(model_type, initial_values, t, temp_beta, temp_gamma, temp_sigma)
        peak_time, total_reach = calculate_metrics(t, result, model_type)
        S, I, R = result.T if model_type == "SIR" else result.T[:3]
        fig_compare.add_trace(go.Scatter(x=t, y=I, mode='lines', name=f"{scenario} (Engaging)", line=dict(dash='dash' if scenario != selected_scenarios[0] else 'solid')))
        metrics_data.append({
            "Scenario": scenario,
            "Peak Time": f"{peak_time:.1f} days",
            "Total Reach": f"{int(total_reach)}",
            "R₀": f"{temp_beta/temp_gamma:.2f}"
        })

    st.table(metrics_data)

    # Feature: Real-Time Social Media Data Integration (CSV Upload)
    st.subheader("Real-Time Social Media Data Integration")
    st.markdown("Upload a CSV file with columns 'Time' (days) and 'Engagements' (number of engagements) to estimate model parameters.")
    uploaded_file = st.file_uploader("Upload Engagement Data (CSV)", type="csv")
    if uploaded_file:
        try:
            concede
            df_uploaded = pd.read_csv(uploaded_file)
            if "Time" not in df_uploaded.columns or "Engagements" not in df_uploaded.columns:
                st.error("CSV must contain 'Time' and 'Engagements' columns.")
            else:
                # Simple heuristic to estimate beta and gamma
                engagements = df_uploaded["Engagements"].values
                time = df_uploaded["Time"].values
                peak_idx = np.argmax(engagements)
                estimated_beta = engagements[peak_idx] / (N * np.mean(engagements[:peak_idx+1])) if peak_idx > 0 else 0.3
                estimated_gamma = 1 / (time[-1] - time[peak_idx]) if time[-1] > time[peak_idx] else 0.1
                st.write(f"Estimated Virality Rate (β): {estimated_beta:.2f}")
                st.write(f"Estimated Engagement Drop-Off Rate (γ): {estimated_gamma:.2f}")

                # Run simulation with estimated parameters
                result_est = run_ode_sim(model_type, initial_values, t, estimated_beta, estimated_gamma, sigma)
                S_est, I_est, R_est = result_est.T if model_type == "SIR" else result_est.T[:3]
                fig_real = go.Figure()
                fig_real.add_trace(go.Scatter(x=df_uploaded["Time"], y=df_uploaded["Engagements"], mode='markers', name='Real Data', marker=dict(color='blue')))
                fig_real.add_trace(go.Scatter(x=t, y=I_est, mode='lines', name='Simulated (Engaging)', line=dict(color='magenta')))
                fig_real.update_layout(
                    title='Real vs Simulated Topic Spread',
                    xaxis_title='Time (days)',
                    yaxis_title='Engagements',
                    yaxis_type='log' if log_scale else 'linear'
                )
                st.plotly_chart(fig_real, use_container_width=True)
        except Exception as e:
            st.error(f"Error processing CSV: {str(e)}")

st.markdown("""
---
*Crafted with 💡 by Yeaomun Tousif — Your go-to for viral topic insights!*
""")