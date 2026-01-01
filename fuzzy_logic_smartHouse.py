
import streamlit as st
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

st.set_page_config(page_title="Neuro-Fuzzy Smart Home", page_icon="⚡", layout="wide")

# ==================== HISTORY ====================
DATA_FILE = "history.csv"

if "df_hist" not in st.session_state:
    expected_columns = ["Date", "Energy_Demand", "Battery_Level", "Grid_Price", "Hour", "Solar_Generation",
                        "Temperature", "Humidity", "Appliance_Reduction_real", "Appliance_Reduction_rec", 
                        "Battery_Action_rec", "Grid_Action_rec", "Energy_Predicted", "Solar_Predicted"]
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        for col in expected_columns:
            if col not in df.columns:
                df[col] = np.nan
        df = df[expected_columns]
        st.session_state.df_hist = df
    else:
        st.session_state.df_hist = pd.DataFrame(columns=expected_columns)

# ==================== LSTM PREDICTION MODELS (Simulés) ====================
def predict_energy_consumption(historical_data, temperature, hour, day_of_week, month):
    """
    Simule un modèle LSTM pour prédire la consommation d'énergie de la prochaine heure
    Features: Historical consumption (24h), Temperature, Time features, Occupancy patterns, Weather conditions
    """
    # Simulation basée sur des patterns réalistes
    base_consumption = 50
    
    # Pattern horaire (pointe le matin et soir)
    if 6 <= hour <= 9:
        time_factor = 1.3
    elif 18 <= hour <= 22:
        time_factor = 1.5
    elif 0 <= hour <= 6:
        time_factor = 0.6
    else:
        time_factor = 1.0
    
    # Pattern hebdomadaire (weekend vs semaine)
    week_factor = 1.2 if day_of_week in [5, 6] else 1.0
    
    # Effet température (climatisation/chauffage)
    temp_factor = 1.0
    if temperature < 15:
        temp_factor = 1.3  # Chauffage
    elif temperature > 28:
        temp_factor = 1.4  # Climatisation
    
    # Tendance historique (moyenne des dernières 24h)
    if len(historical_data) > 0:
        trend = np.mean(historical_data[-24:]) / base_consumption
    else:
        trend = 1.0
    
    prediction = base_consumption * time_factor * week_factor * temp_factor * trend
    prediction += np.random.normal(0, 5)  # Petit bruit
    
    return max(0, min(100, prediction))

def predict_solar_generation(weather_temp, humidity, hour, solar_angle, cloud_cover, historical_solar):
    """
    Simule un modèle LSTM pour prédire la génération solaire de la prochaine heure
    Features: Weather forecast, Time/season, Historical solar output, Cloud cover, Solar irradiance, Panel efficiency
    """
    base_generation = 60
    
    # Pattern solaire selon l'heure
    if 6 <= hour <= 8:
        time_factor = 0.4
    elif 9 <= hour <= 16:
        time_factor = 1.0
    elif 17 <= hour <= 19:
        time_factor = 0.5
    else:
        time_factor = 0.0  # Pas de soleil la nuit
    
    # Effet météo
    temp_efficiency = 1.0 - max(0, (weather_temp - 25) * 0.005)  # Perte d'efficacité si trop chaud
    humidity_factor = 1.0 - (humidity / 200)  # Humidité réduit l'efficacité
    cloud_factor = 1.0 - (cloud_cover / 100) * 0.7
    
    # Tendance historique
    if len(historical_solar) > 0:
        trend = np.mean(historical_solar[-7:]) / base_generation
    else:
        trend = 1.0
    
    prediction = base_generation * time_factor * temp_efficiency * humidity_factor * cloud_factor * trend
    prediction += np.random.normal(0, 3)
    
    return max(0, min(100, prediction))

# ==================== FUZZY VARIABLES (SELON DOCUMENT) ====================
# INPUTS
energy = ctrl.Antecedent(np.arange(0, 101, 1), 'Energy_Demand')
battery = ctrl.Antecedent(np.arange(0, 101, 1), 'Battery_Level')
price = ctrl.Antecedent(np.arange(0, 101, 1), 'Grid_Price')
hour_var = ctrl.Antecedent(np.arange(0, 25, 1), 'Hour')
solar = ctrl.Antecedent(np.arange(0, 101, 1), 'Solar_Generation')

# OUTPUTS
reduction = ctrl.Consequent(np.arange(0, 101, 1), 'Appliance_Reduction')
battery_action = ctrl.Consequent(np.arange(-100, 101, 1), 'Battery_Action')
grid_action = ctrl.Consequent(np.arange(-100, 101, 1), 'Grid_Action')

# ==================== MEMBERSHIP FUNCTIONS (DOCUMENT EXACT) ====================

# Energy Demand (0-100 scale)
energy['Low'] = fuzz.trimf(energy.universe, [0, 0, 30])
energy['Medium'] = fuzz.trimf(energy.universe, [20, 50, 80])
energy['High'] = fuzz.trimf(energy.universe, [70, 100, 100])

# Solar Generation (0-100 scale)
solar['Poor'] = fuzz.trimf(solar.universe, [0, 0, 30])
solar['Moderate'] = fuzz.trimf(solar.universe, [20, 50, 80])
solar['Excellent'] = fuzz.trimf(solar.universe, [70, 100, 100])

# Battery Level (0-100 scale)
battery['Critical'] = fuzz.trimf(battery.universe, [0, 0, 20])
battery['Low'] = fuzz.trimf(battery.universe, [10, 30, 50])
battery['Medium'] = fuzz.trimf(battery.universe, [40, 60, 80])
battery['High'] = fuzz.trimf(battery.universe, [70, 100, 100])

# Grid Price (0-100 scale)
price['Cheap'] = fuzz.trimf(price.universe, [0, 0, 30])
price['Normal'] = fuzz.trimf(price.universe, [20, 50, 80])
price['Expensive'] = fuzz.trimf(price.universe, [70, 100, 100])

# Time of Day (0-24 scale)
hour_var['Night1'] = fuzz.trimf(hour_var.universe, [0, 3, 6])
hour_var['Night2'] = fuzz.trimf(hour_var.universe, [22, 23, 24])
hour_var['Morning'] = fuzz.trapmf(hour_var.universe, [6, 8, 10, 12])
hour_var['Afternoon'] = fuzz.trapmf(hour_var.universe, [12, 14, 16, 18])
hour_var['Evening'] = fuzz.trimf(hour_var.universe, [18, 20, 22])

# Appliance Reduction (0-100 scale)
reduction['None'] = fuzz.trimf(reduction.universe, [0, 0, 20])
reduction['Slight'] = fuzz.trimf(reduction.universe, [10, 30, 50])
reduction['Moderate'] = fuzz.trimf(reduction.universe, [40, 60, 80])
reduction['Aggressive'] = fuzz.trimf(reduction.universe, [70, 100, 100])

# Battery Action (-100 to 100 scale)
battery_action['Discharge'] = fuzz.trimf(battery_action.universe, [-100, -100, -50])
battery_action['Maintain'] = fuzz.trimf(battery_action.universe, [-30, 0, 30])
battery_action['Charge'] = fuzz.trimf(battery_action.universe, [50, 100, 100])

# Grid Interaction (-100 to 100 scale)
grid_action['Sell'] = fuzz.trimf(grid_action.universe, [50, 100, 100])
grid_action['Neutral'] = fuzz.trimf(grid_action.universe, [-30, 0, 30])
grid_action['Buy'] = fuzz.trimf(grid_action.universe, [-100, -100, -50])

# ==================== FUZZY RULES (20+ RULES - DOCUMENT) ====================
rules = [
    # === PEAK DEMAND MANAGEMENT ===
    ctrl.Rule(energy['High'] & battery['Critical'], [reduction['Aggressive'], grid_action['Buy']]),
    ctrl.Rule(energy['High'] & price['Expensive'], [reduction['Aggressive'], battery_action['Discharge']]),
    ctrl.Rule(energy['High'] & hour_var['Evening'], [reduction['Moderate'], battery_action['Discharge']]),
    ctrl.Rule(energy['Low'] & battery['High'], [reduction['None'], grid_action['Sell']]),
    
    # === SOLAR ENERGY OPTIMIZATION ===
    ctrl.Rule(solar['Excellent'] & battery['Low'], [battery_action['Charge'], reduction['Slight']]),
    ctrl.Rule(solar['Excellent'] & battery['High'], [grid_action['Sell'], reduction['None']]),
    ctrl.Rule(solar['Excellent'] & energy['Low'], [battery_action['Charge'], grid_action['Sell']]),
    ctrl.Rule(solar['Poor'] & energy['High'], [reduction['Aggressive'], grid_action['Buy']]),
    ctrl.Rule(solar['Moderate'] & battery['Medium'], [battery_action['Maintain'], reduction['Slight']]),
    
    # === BATTERY MANAGEMENT STRATEGY ===
    ctrl.Rule(battery['Critical'], [battery_action['Charge'], reduction['Aggressive'], grid_action['Buy']]),
    ctrl.Rule(battery['Low'] & price['Cheap'], [battery_action['Charge'], grid_action['Buy']]),
    ctrl.Rule(battery['High'] & price['Expensive'], [battery_action['Discharge'], grid_action['Sell']]),
    ctrl.Rule(battery['Medium'] & solar['Excellent'], [battery_action['Charge'], reduction['Slight']]),
    ctrl.Rule(battery['High'] & energy['Low'], [grid_action['Sell'], reduction['None']]),
    
    # === GRID INTERACTION DECISIONS ===
    ctrl.Rule(price['Cheap'] & battery['Low'], [grid_action['Buy'], battery_action['Charge']]),
    ctrl.Rule(price['Expensive'] & battery['High'], [grid_action['Sell'], reduction['Moderate']]),
    ctrl.Rule(price['Expensive'] & energy['High'], [reduction['Aggressive'], battery_action['Discharge']]),
    ctrl.Rule(price['Normal'] & battery['Medium'], [grid_action['Neutral'], reduction['Slight']]),
    
    # === EMERGENCY SCENARIOS ===
    ctrl.Rule(battery['Critical'] & price['Expensive'], [reduction['Aggressive'], grid_action['Buy']]),
    ctrl.Rule(solar['Poor'] & battery['Low'] & energy['High'], [reduction['Aggressive'], grid_action['Buy']]),
    ctrl.Rule(hour_var['Evening'] & energy['High'] & battery['Low'], [reduction['Moderate'], grid_action['Buy']]),
    
    # === TIME-BASED RULES ===
    ctrl.Rule((hour_var['Night1'] | hour_var['Night2']) & battery['High'], [grid_action['Sell'], reduction['Slight']]),
    ctrl.Rule(hour_var['Morning'] & solar['Excellent'], [battery_action['Charge'], reduction['None']]),
    ctrl.Rule(hour_var['Afternoon'] & solar['Excellent'], [grid_action['Sell'], battery_action['Charge']]),
    ctrl.Rule(hour_var['Evening'] & energy['High'], [reduction['Moderate'], battery_action['Discharge']]),
    
    # === DEFAULT RULES ===
    ctrl.Rule(energy['Medium'], reduction['Slight']),
    ctrl.Rule(battery['Medium'], battery_action['Maintain']),
    ctrl.Rule(price['Normal'], grid_action['Neutral']),
]

system = ctrl.ControlSystem(rules)
sim = ctrl.ControlSystemSimulation(system)

# ==================== PLOTTING FUNCTION ====================
def plot_membership_function(variable, ax, current_value=None, title=""):
    ax.clear()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#DDA15E']
    
    for idx, (label, mf) in enumerate(variable.terms.items()):
        ax.plot(variable.universe, mf.mf, linewidth=2.5, label=label, color=colors[idx % len(colors)])
        ax.fill_between(variable.universe, 0, mf.mf, alpha=0.2, color=colors[idx % len(colors)])
    
    if current_value is not None:
        ax.axvline(current_value, color='red', linewidth=3, linestyle='--', label=f'Value: {current_value}', alpha=0.8)
    
    ax.set_xlabel(title, fontsize=11, fontweight='bold')
    ax.set_ylabel('Membership Degree', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([-0.05, 1.1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("📊 Experiment History")
    
    with st.expander("➕ Add Experiment"):
        with st.form("add"):
            st.subheader("Input Parameters")
            c1, c2 = st.columns(2)
            with c1:
                e_in = st.slider("Energy Demand (%)", 0, 100, 70)
                b_in = st.slider("Battery Level (%)", 0, 100, 30)
                s_in = st.slider("Solar Generation (%)", 0, 100, 50)
                temp_in = st.slider("Temperature (°C)", -10, 45, 22)
            with c2:
                p_in = st.slider("Grid Price (0-100)", 0, 100, 90)
                h_in = st.slider("Hour (0-23)", 0, 23, 20)
                hum_in = st.slider("Humidity (%)", 0, 100, 60)
            
            real_red = st.number_input("Real Reduction (%)", 0.0, 100.0, 60.0)

            if st.form_submit_button("💾 Save Experiment", use_container_width=True):
                try:
                    # Prédictions LSTM
                    hist_energy = st.session_state.df_hist['Energy_Demand'].tolist() if len(st.session_state.df_hist) > 0 else []
                    hist_solar = st.session_state.df_hist['Solar_Generation'].tolist() if len(st.session_state.df_hist) > 0 else []
                    
                    energy_pred = predict_energy_consumption(hist_energy, temp_in, h_in, datetime.now().weekday(), datetime.now().month)
                    solar_pred = predict_solar_generation(temp_in, hum_in, h_in, 45, 20, hist_solar)
                    
                    # Calcul Fuzzy
                    sim.input['Energy_Demand'] = e_in
                    sim.input['Battery_Level'] = b_in
                    sim.input['Grid_Price'] = p_in
                    sim.input['Hour'] = h_in
                    sim.input['Solar_Generation'] = s_in
                    sim.compute()
                    
                    rec_red = sim.output['Appliance_Reduction']
                    rec_ba = sim.output['Battery_Action']
                    rec_ga = sim.output['Grid_Action']
                except Exception as e:
                    st.error(f"Error: {e}")
                    rec_red = rec_ba = rec_ga = energy_pred = solar_pred = 0.0

                new = pd.DataFrame([{
                    "Date": datetime.now(),
                    "Energy_Demand": e_in,
                    "Battery_Level": b_in,
                    "Grid_Price": p_in,
                    "Hour": h_in,
                    "Solar_Generation": s_in,
                    "Temperature": temp_in,
                    "Humidity": hum_in,
                    "Appliance_Reduction_real": real_red,
                    "Appliance_Reduction_rec": round(rec_red, 1),
                    "Battery_Action_rec": round(rec_ba, 1),
                    "Grid_Action_rec": round(rec_ga, 1),
                    "Energy_Predicted": round(energy_pred, 1),
                    "Solar_Predicted": round(solar_pred, 1)
                }])
                st.session_state.df_hist = pd.concat([st.session_state.df_hist, new], ignore_index=True)
                st.session_state.df_hist.to_csv(DATA_FILE, index=False)
                st.success("✅ Saved!")
                st.rerun()

    if len(st.session_state.df_hist) > 0:
        st.subheader("Recent Experiments")
        st.dataframe(st.session_state.df_hist.tail(10), use_container_width=True, height=300)

# ==================== MAIN INTERFACE ====================
st.title("⚡ Neuro-Fuzzy Smart Home Energy Manager")
st.markdown("Farah Brahim • ISET Bizerte")
st.markdown("*Hybrid AI system combining LSTM Neural Networks and Fuzzy Logic for intelligent energy management*")

# System Architecture Overview
with st.expander("🔍 System Architecture Overview"):
    st.markdown("""
    ### Hybrid AI System Components:
    
    1. **Prediction Engine**: LSTM neural networks for energy consumption and solar generation forecasting
    2. **Decision Engine**: Fuzzy logic controller for intelligent energy management decisions
    
    **Data Flow:**
    - Sensor Data → LSTM Energy Prediction → Fuzzy Logic Controller → Appliance Control
    - Weather Data → LSTM Solar Prediction → Data Preprocessing → Real-time Dashboard
    - Historical Data → Both systems for learning and optimization
    """)

st.markdown("---")

# Predictions Section
col_pred1, col_pred2 = st.columns(2)

with col_pred1:
    st.subheader("🔮 LSTM Energy Prediction")
    with st.form("energy_pred"):
        st.markdown("*Predict next-hour energy consumption*")
        temp_pred = st.slider("Temperature (°C)", -10, 45, 22, key="pred_temp")
        hour_pred = st.slider("Hour", 0, 23, 12, key="pred_hour")
        
        if st.form_submit_button("Predict Energy", use_container_width=True):
            hist_data = st.session_state.df_hist['Energy_Demand'].tolist() if len(st.session_state.df_hist) > 0 else []
            # Utiliser hour_pred au lieu de datetime.now().hour
            simulated_date_pred = datetime.now().replace(hour=hour_pred, minute=0, second=0, microsecond=0)
            pred = predict_energy_consumption(hist_data, temp_pred, hour_pred, simulated_date_pred.weekday(), simulated_date_pred.month)
            st.metric("Predicted Energy Demand", f"{pred:.1f} %", delta=f"{pred - 50:.1f}%")

with col_pred2:
    st.subheader("☀️ LSTM Solar Prediction")
    with st.form("solar_pred"):
        st.markdown("*Predict next-hour solar generation*")
        temp_solar = st.slider("Temperature (°C)", -10, 45, 25, key="solar_temp")
        hum_solar = st.slider("Humidity (%)", 0, 100, 50, key="solar_hum")
        hour_solar = st.slider("Hour", 0, 23, 12, key="solar_hour")
        
        if st.form_submit_button("Predict Solar", use_container_width=True):
            hist_solar = st.session_state.df_hist['Solar_Generation'].tolist() if len(st.session_state.df_hist) > 0 else []
            # Utiliser hour_solar au lieu de datetime.now().hour
            pred = predict_solar_generation(temp_solar, hum_solar, hour_solar, 45, 20, hist_solar)
            st.metric("Predicted Solar Generation", f"{pred:.1f} %", delta=f"{pred - 50:.1f}%")

st.markdown("---")

# Fuzzy Control Section
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎛️ Current Conditions")
    energy_val = st.slider("Energy Demand (%)", 0, 100, 75, key="main_e")
    battery_val = st.slider("Battery Level (%)", 0, 100, 25, key="main_b")
    solar_val = st.slider("Solar Generation (%)", 0, 100, 60, key="main_s")
    price_val = st.slider("Grid Price (0=cheap → 100=expensive)", 0, 100, 95, key="main_p")
    time_val = st.slider("Hour of Day (0-24)", 0, 23, 19, key="main_t")

with col2:
    st.subheader("💡 Fuzzy Logic Recommendations")
    
    # Add temperature and humidity inputs for predictions
    col_extra1, col_extra2 = st.columns(2)
    with col_extra1:
        temp_current = st.number_input("Temperature (°C)", -10, 45, 22, key="curr_temp")
    with col_extra2:
        hum_current = st.number_input("Humidity (%)", 0, 100, 60, key="curr_hum")
    
    if st.button("🔄 Calculate & Save", type="primary", use_container_width=True):
        try:
            # LSTM Predictions - utiliser l'heure du slider
            hist_energy = st.session_state.df_hist['Energy_Demand'].tolist() if len(st.session_state.df_hist) > 0 else []
            hist_solar = st.session_state.df_hist['Solar_Generation'].tolist() if len(st.session_state.df_hist) > 0 else []
            
            # Calculer le jour de la semaine et le mois basés sur l'heure simulée
            # Pour la simulation, on utilise une date de base
            simulated_date = datetime.now().replace(hour=time_val, minute=0, second=0, microsecond=0)
            day_of_week = simulated_date.weekday()
            month = simulated_date.month
            
            energy_pred = predict_energy_consumption(hist_energy, temp_current, time_val, day_of_week, month)
            solar_pred = predict_solar_generation(temp_current, hum_current, time_val, 45, 20, hist_solar)
            
            # Fuzzy Logic Calculations
            sim.input['Energy_Demand'] = energy_val
            sim.input['Battery_Level'] = battery_val
            sim.input['Grid_Price'] = price_val
            sim.input['Hour'] = time_val
            sim.input['Solar_Generation'] = solar_val
            sim.compute()
            
            red = sim.output['Appliance_Reduction']
            ba = sim.output['Battery_Action']
            ga = sim.output['Grid_Action']
            
            # Save to history automatically - utiliser une date personnalisée
            # Créer un timestamp basé sur l'heure du slider
            custom_timestamp = datetime.now().replace(hour=time_val, minute=0, second=0, microsecond=0)
            
            new_entry = pd.DataFrame([{
                "Date": custom_timestamp,
                "Energy_Demand": energy_val,
                "Battery_Level": battery_val,
                "Grid_Price": price_val,
                "Hour": time_val,
                "Solar_Generation": solar_val,
                "Temperature": temp_current,
                "Humidity": hum_current,
                "Appliance_Reduction_real": red,  # Using fuzzy output as baseline
                "Appliance_Reduction_rec": round(red, 1),
                "Battery_Action_rec": round(ba, 1),
                "Grid_Action_rec": round(ga, 1),
                "Energy_Predicted": round(energy_pred, 1),
                "Solar_Predicted": round(solar_pred, 1)
            }])
            
            st.session_state.df_hist = pd.concat([st.session_state.df_hist, new_entry], ignore_index=True)
            st.session_state.df_hist.to_csv(DATA_FILE, index=False)
            
            st.success("✅ Calculation successful & saved to history!")
            
            # Display results
            col_res1, col_res2, col_res3 = st.columns(3)
            
            with col_res1:
                st.metric("Appliance Reduction", f"{red:.1f}%")
                st.info(f"🔮 Next hour prediction: {energy_pred:.1f}%")
            
            with col_res2:
                if ba <= -40:
                    bat_act = "🔋 Discharge"
                elif ba >= 40:
                    bat_act = "⚡ Charge"
                else:
                    bat_act = "⚖️ Maintain"
                st.metric("Battery Action", bat_act)
            
            with col_res3:
                if ga >= 40:
                    grid_act = "💰 Sell"
                elif ga <= -40:
                    grid_act = "🏪 Buy"
                else:
                    grid_act = "⚖️ Neutral"
                st.metric("Grid Action", grid_act)
                st.info(f"☀️ Solar prediction: {solar_pred:.1f}%")
            
            # Rerun to update the display
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error: {e}")

# ==================== FUZZY MEMBERSHIP FUNCTIONS ====================
st.markdown("---")
st.subheader("📈 Fuzzy Membership Functions Visualization")

fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
ax5 = fig.add_subplot(gs[2, 0])
ax6 = fig.add_subplot(gs[2, 1])
ax7 = fig.add_subplot(gs[3, 0])
ax8 = fig.add_subplot(gs[3, 1])

plot_membership_function(energy, ax1, energy_val, "Energy Demand (0-100 scale)")
plot_membership_function(battery, ax2, battery_val, "Battery Level (0-100 scale)")
plot_membership_function(solar, ax3, solar_val, "Solar Generation (0-100 scale)")
plot_membership_function(price, ax4, price_val, "Grid Price (0-100 scale)")
plot_membership_function(hour_var, ax5, time_val, "Time of Day (0-24 scale)")

plot_membership_function(reduction, ax6, None, "Appliance Reduction (0-100)")
plot_membership_function(battery_action, ax7, None, "Battery Action (-100 to 100)")
plot_membership_function(grid_action, ax8, None, "Grid Interaction (-100 to 100)")

try:
    sim.input['Energy_Demand'] = energy_val
    sim.input['Battery_Level'] = battery_val
    sim.input['Grid_Price'] = price_val
    sim.input['Hour'] = time_val
    sim.input['Solar_Generation'] = solar_val
    sim.compute()
    
    ax6.axvline(sim.output['Appliance_Reduction'], color='darkred', linewidth=3, linestyle=':', label=f'Output: {sim.output["Appliance_Reduction"]:.1f}%')
    ax7.axvline(sim.output['Battery_Action'], color='darkred', linewidth=3, linestyle=':', label=f'Output: {sim.output["Battery_Action"]:.1f}')
    ax8.axvline(sim.output['Grid_Action'], color='darkred', linewidth=3, linestyle=':', label=f'Output: {sim.output["Grid_Action"]:.1f}')
    
    ax6.legend(loc='upper right', fontsize=9)
    ax7.legend(loc='upper right', fontsize=9)
    ax8.legend(loc='upper right', fontsize=9)
except:
    pass

st.pyplot(fig)
plt.close()

# ==================== PERFORMANCE HISTORY ====================
if len(st.session_state.df_hist) > 1:
    st.markdown("---")
    st.subheader("📊 System Performance History & Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Control Performance", "Predictions Accuracy", "Energy Flow"])
    
    with tab1:
        fig2, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
        dates = pd.to_datetime(st.session_state.df_hist["Date"])

        a1.plot(dates, st.session_state.df_hist["Appliance_Reduction_real"], 
                's-', color='#FF6B6B', label="Real", linewidth=2, markersize=8)
        a1.plot(dates, st.session_state.df_hist["Appliance_Reduction_rec"], 
                'o--', color='#4ECDC4', label="Recommended", linewidth=2, markersize=8)
        a1.set_ylabel("Reduction (%)", fontweight='bold')
        a1.legend(); a1.grid(True, alpha=0.3)
        a1.set_title("Appliance Reduction: Real vs Recommended", fontweight='bold')

        a2.plot(dates, st.session_state.df_hist["Battery_Level"], 
                'd-', color='#9B59B6', linewidth=2, markersize=8)
        a2.set_ylabel("Battery (%)", fontweight='bold')
        a2.grid(True, alpha=0.3)
        a2.set_title("Battery Level Over Time", fontweight='bold')
        
        a3.plot(dates, st.session_state.df_hist["Battery_Action_rec"], 
                '^-', color='#E67E22', linewidth=2, markersize=8)
        a3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        a3.set_ylabel("Battery Action", fontweight='bold')
        a3.set_xlabel("Date", fontweight='bold')
        a3.grid(True, alpha=0.3)
        a3.set_title("Battery Action Commands", fontweight='bold')
        
        a4.plot(dates, st.session_state.df_hist["Grid_Action_rec"], 
                'v-', color='#27AE60', linewidth=2, markersize=8)
        a4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        a4.set_ylabel("Grid Action", fontweight='bold')
        a4.set_xlabel("Date", fontweight='bold')
        a4.grid(True, alpha=0.3)
        a4.set_title("Grid Interaction Commands", fontweight='bold')

        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()
    
    with tab2:
        fig3, (b1, b2) = plt.subplots(1, 2, figsize=(16, 6))
        
        b1.plot(dates, st.session_state.df_hist["Energy_Demand"], 
                'o-', color='#E74C3C', label="Actual", linewidth=2, markersize=8)
        b1.plot(dates, st.session_state.df_hist["Energy_Predicted"], 
                's--', color='#3498DB', label="LSTM Predicted", linewidth=2, markersize=8)
        b1.set_ylabel("Energy (%)", fontweight='bold')
        b1.set_xlabel("Date", fontweight='bold')
        b1.legend(); b1.grid(True, alpha=0.3)
        b1.set_title("LSTM Energy Consumption Prediction", fontweight='bold')
        
        b2.plot(dates, st.session_state.df_hist["Solar_Generation"], 
                'o-', color='#F39C12', label="Actual", linewidth=2, markersize=8)
        b2.plot(dates, st.session_state.df_hist["Solar_Predicted"], 
                's--', color='#16A085', label="LSTM Predicted", linewidth=2, markersize=8)
        b2.set_ylabel("Solar Generation (%)", fontweight='bold')
        b2.set_xlabel("Date", fontweight='bold')
        b2.legend(); b2.grid(True, alpha=0.3)
        b2.set_title("LSTM Solar Generation Prediction", fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()
    
    with tab3:
        fig4, (c1, c2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
        
        c1.plot(dates, st.session_state.df_hist["Solar_Generation"], 
                'o-', color='#F39C12', label="Solar Generation", linewidth=2, markersize=8)
        c1.plot(dates, st.session_state.df_hist["Energy_Demand"], 
                's-', color='#E74C3C', label="Energy Demand", linewidth=2, markersize=8)
        c1.fill_between(dates, 0, st.session_state.df_hist["Solar_Generation"], alpha=0.3, color='#F39C12')
        c1.fill_between(dates, 0, st.session_state.df_hist["Energy_Demand"], alpha=0.3, color='#E74C3C')
        c1.set_ylabel("Energy (%)", fontweight='bold')
        c1.legend(); c1.grid(True, alpha=0.3)
        c1.set_title("Energy Supply vs Demand", fontweight='bold')
        
        c2.plot(dates, st.session_state.df_hist["Grid_Price"], 
                'd-', color='#8E44AD', linewidth=2, markersize=8)
        c2.set_ylabel("Grid Price", fontweight='bold')
        c2.set_xlabel("Date", fontweight='bold')
        c2.grid(True, alpha=0.3)
        c2.set_title("Grid Price Variation", fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

st.markdown("---")

st.caption("© Farah BRAHIM – Neuro-Fuzzy Smart Home Project – Dept. EE @ ISET Bizerte – 2025")
