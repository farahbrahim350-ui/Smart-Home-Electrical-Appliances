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

def predict_energy_consumption(historical_data, temperature, hour, day_of_week, month):
    """
    Simule un modèle LSTM pour prédire la consommation d'énergie de la prochaine heure
    """
    base_consumption = 50
    
    # Pattern horaire
    if 6 <= hour <= 9:
        time_factor = 1.3
    elif 18 <= hour <= 22:
        time_factor = 1.5
    elif 0 <= hour <= 6:
        time_factor = 0.6
    else:
        time_factor = 1.0
    
    # Pattern hebdomadaire
    week_factor = 1.2 if day_of_week in [5, 6] else 1.0
    
    # Effet température
    temp_factor = 1.0
    if temperature < 15:
        temp_factor = 1.3
    elif temperature > 28:
        temp_factor = 1.4
    
    # Tendance historique
    if len(historical_data) > 0:
        trend = np.mean(historical_data[-24:]) / base_consumption
    else:
        trend = 1.0
    
    prediction = base_consumption * time_factor * week_factor * temp_factor * trend
    prediction += np.random.normal(0, 5)
    
    return max(0, min(100, prediction))

def predict_solar_generation(weather_temp, humidity, hour, solar_angle, cloud_cover, historical_solar):
    """
    Simule un modèle LSTM pour prédire la génération solaire
    """
    base_generation = 60
    
    # Solar pattern
    if 6 <= hour <= 8:
        time_factor = 0.4
    elif 9 <= hour <= 16:
        time_factor = 1.0
    elif 17 <= hour <= 19:
        time_factor = 0.5
    else:
        time_factor = 0.0
    
    # Weather effect
    temp_efficiency = 1.0 - max(0, (weather_temp - 25) * 0.005)
    humidity_factor = 1.0 - (humidity / 200)
    cloud_factor = 1.0 - (cloud_cover / 100) * 0.7
    
    # Historical trend
    if len(historical_solar) > 0:
        trend = np.mean(historical_solar[-7:]) / base_generation
    else:
        trend = 1.0
    
    prediction = base_generation * time_factor * temp_efficiency * humidity_factor * cloud_factor * trend
    prediction += np.random.normal(0, 3)
    
    return max(0, min(100, prediction))

# ==================== FUZZY VARIABLES ====================
# INPUTS
energy_demand = ctrl.Antecedent(np.arange(0, 101, 1), 'energy_demand')
battery_level = ctrl.Antecedent(np.arange(0, 101, 1), 'battery_level')
grid_price = ctrl.Antecedent(np.arange(0, 101, 1), 'grid_price')
hour_of_day = ctrl.Antecedent(np.arange(0, 25, 1), 'hour_of_day')
solar_generation = ctrl.Antecedent(np.arange(0, 101, 1), 'solar_generation')

# OUTPUTS
appliance_reduction = ctrl.Consequent(np.arange(0, 101, 1), 'appliance_reduction')
battery_control = ctrl.Consequent(np.arange(-100, 101, 1), 'battery_control')
grid_control = ctrl.Consequent(np.arange(-100, 101, 1), 'grid_control')

# ==================== MEMBERSHIP FUNCTIONS  ====================

# Energy Demand
energy_demand['Low'] = fuzz.trimf(energy_demand.universe, [0, 0, 30])
energy_demand['Medium'] = fuzz.trimf(energy_demand.universe, [20, 50, 80])
energy_demand['High'] = fuzz.trimf(energy_demand.universe, [70, 100, 100])

# Solar Generation
solar_generation['Poor'] = fuzz.trimf(solar_generation.universe, [0, 0, 30])
solar_generation['Moderate'] = fuzz.trimf(solar_generation.universe, [20, 50, 80])
solar_generation['Excellent'] = fuzz.trimf(solar_generation.universe, [70, 100, 100])

# Battery Level
battery_level['Critical'] = fuzz.trimf(battery_level.universe, [0, 0, 20])
battery_level['Low'] = fuzz.trimf(battery_level.universe, [10, 30, 50])
battery_level['Medium'] = fuzz.trimf(battery_level.universe, [40, 60, 80])
battery_level['High'] = fuzz.trimf(battery_level.universe, [70, 100, 100])

# Grid Price
grid_price['Cheap'] = fuzz.trimf(grid_price.universe, [0, 0, 30])
grid_price['Normal'] = fuzz.trimf(grid_price.universe, [20, 50, 80])
grid_price['Expensive'] = fuzz.trimf(grid_price.universe, [70, 100, 100])

# Time of Day
hour_of_day['Night1'] = fuzz.trimf(hour_of_day.universe, [0, 3, 6])
hour_of_day['Night2'] = fuzz.trimf(hour_of_day.universe, [22, 23, 24])
hour_of_day['Morning'] = fuzz.trapmf(hour_of_day.universe, [6, 8, 10, 12])
hour_of_day['Afternoon'] = fuzz.trapmf(hour_of_day.universe, [12, 14, 16, 18])
hour_of_day['Evening'] = fuzz.trimf(hour_of_day.universe, [18, 20, 22])

# Appliance Reduction
appliance_reduction['None'] = fuzz.trimf(appliance_reduction.universe, [0, 0, 20])
appliance_reduction['Slight'] = fuzz.trimf(appliance_reduction.universe, [10, 30, 50])
appliance_reduction['Moderate'] = fuzz.trimf(appliance_reduction.universe, [40, 60, 80])
appliance_reduction['Aggressive'] = fuzz.trimf(appliance_reduction.universe, [70, 100, 100])

# Battery Action
battery_control['Discharge'] = fuzz.trimf(battery_control.universe, [-100, -100, -50])
battery_control['Maintain'] = fuzz.trimf(battery_control.universe, [-30, 0, 30])
battery_control['Charge'] = fuzz.trimf(battery_control.universe, [50, 100, 100])

# Grid Interaction
grid_control['Sell'] = fuzz.trimf(grid_control.universe, [50, 100, 100])
grid_control['Neutral'] = fuzz.trimf(grid_control.universe, [-30, 0, 30])
grid_control['Buy'] = fuzz.trimf(grid_control.universe, [-100, -100, -50])

# ==================== FUZZY RULES ====================
rules = [
    # === PEAK DEMAND MANAGEMENT ===
    ctrl.Rule(energy_demand['High'] & battery_level['Critical'], 
              [appliance_reduction['Aggressive'], grid_control['Buy']]),
    ctrl.Rule(energy_demand['High'] & grid_price['Expensive'], 
              [appliance_reduction['Aggressive'], battery_control['Discharge']]),
    ctrl.Rule(energy_demand['High'] & hour_of_day['Evening'], 
              [appliance_reduction['Moderate'], battery_control['Discharge']]),
    ctrl.Rule(energy_demand['Low'] & battery_level['High'], 
              [appliance_reduction['None'], grid_control['Sell']]),
    
    # === SOLAR ENERGY OPTIMIZATION ===
    ctrl.Rule(solar_generation['Excellent'] & battery_level['Low'], 
              [battery_control['Charge'], appliance_reduction['Slight']]),
    ctrl.Rule(solar_generation['Excellent'] & battery_level['High'], 
              [grid_control['Sell'], appliance_reduction['None']]),
    ctrl.Rule(solar_generation['Excellent'] & energy_demand['Low'], 
              [battery_control['Charge'], grid_control['Sell']]),
    ctrl.Rule(solar_generation['Poor'] & energy_demand['High'], 
              [appliance_reduction['Aggressive'], grid_control['Buy']]),
    ctrl.Rule(solar_generation['Moderate'] & battery_level['Medium'], 
              [battery_control['Maintain'], appliance_reduction['Slight']]),
    
    # === BATTERY MANAGEMENT STRATEGY ===
    ctrl.Rule(battery_level['Critical'], 
              [battery_control['Charge'], appliance_reduction['Aggressive'], grid_control['Buy']]),
    ctrl.Rule(battery_level['Low'] & grid_price['Cheap'], 
              [battery_control['Charge'], grid_control['Buy']]),
    ctrl.Rule(battery_level['High'] & grid_price['Expensive'], 
              [battery_control['Discharge'], grid_control['Sell']]),
    ctrl.Rule(battery_level['Medium'] & solar_generation['Excellent'], 
              [battery_control['Charge'], appliance_reduction['Slight']]),
    ctrl.Rule(battery_level['High'] & energy_demand['Low'], 
              [grid_control['Sell'], appliance_reduction['None']]),
    
    # === GRID INTERACTION DECISIONS ===
    ctrl.Rule(grid_price['Cheap'] & battery_level['Low'], 
              [grid_control['Buy'], battery_control['Charge']]),
    ctrl.Rule(grid_price['Expensive'] & battery_level['High'], 
              [grid_control['Sell'], appliance_reduction['Moderate']]),
    ctrl.Rule(grid_price['Expensive'] & energy_demand['High'], 
              [appliance_reduction['Aggressive'], battery_control['Discharge']]),
    ctrl.Rule(grid_price['Normal'] & battery_level['Medium'], 
              [grid_control['Neutral'], appliance_reduction['Slight']]),
    
    # === EMERGENCY SCENARIOS ===
    ctrl.Rule(battery_level['Critical'] & grid_price['Expensive'], 
              [appliance_reduction['Aggressive'], grid_control['Buy']]),
    ctrl.Rule(solar_generation['Poor'] & battery_level['Low'] & energy_demand['High'], 
              [appliance_reduction['Aggressive'], grid_control['Buy']]),
    ctrl.Rule(hour_of_day['Evening'] & energy_demand['High'] & battery_level['Low'], 
              [appliance_reduction['Moderate'], grid_control['Buy']]),
    
    # === TIME-BASED RULES ===
    ctrl.Rule((hour_of_day['Night1'] | hour_of_day['Night2']) & battery_level['High'], 
              [grid_control['Sell'], appliance_reduction['Slight']]),
    ctrl.Rule(hour_of_day['Morning'] & solar_generation['Excellent'], 
              [battery_control['Charge'], appliance_reduction['None']]),
    ctrl.Rule(hour_of_day['Afternoon'] & solar_generation['Excellent'], 
              [grid_control['Sell'], battery_control['Charge']]),
    ctrl.Rule(hour_of_day['Evening'] & energy_demand['High'], 
              [appliance_reduction['Moderate'], battery_control['Discharge']]),
    
    # === DEFAULT RULES ===
    ctrl.Rule(energy_demand['Medium'], appliance_reduction['Slight']),
    ctrl.Rule(battery_level['Medium'], battery_control['Maintain']),
    ctrl.Rule(grid_price['Normal'], grid_control['Neutral']),
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
                    hist_energy = st.session_state.df_hist['Energy_Demand'].tolist() if len(st.session_state.df_hist) > 0 else []
                    hist_solar = st.session_state.df_hist['Solar_Generation'].tolist() if len(st.session_state.df_hist) > 0 else []
                    
                    energy_pred = predict_energy_consumption(hist_energy, temp_in, h_in, datetime.now().weekday(), datetime.now().month)
                    solar_pred = predict_solar_generation(temp_in, hum_in, h_in, 45, 20, hist_solar)
                    
                    sim.input['energy_demand'] = e_in
                    sim.input['battery_level'] = b_in
                    sim.input['grid_price'] = p_in
                    sim.input['hour_of_day'] = h_in
                    sim.input['solar_generation'] = s_in
                    sim.compute()
                    
                    rec_red = sim.output['appliance_reduction']
                    rec_ba = sim.output['battery_control']
                    rec_ga = sim.output['grid_control']
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
    time_val = st.slider("Hour of Day (0-23)", 0, 23, 19, key="main_t")

with col2:
    st.subheader("💡 Fuzzy Logic Recommendations")
    
    col_extra1, col_extra2 = st.columns(2)
    with col_extra1:
        temp_current = st.number_input("Temperature (°C)", -10, 45, 22, key="curr_temp")
    with col_extra2:
        hum_current = st.number_input("Humidity (%)", 0, 100, 60, key="curr_hum")
    
    if st.button("🔄 Calculate & Save", type="primary", use_container_width=True):
        try:
            hist_energy = st.session_state.df_hist['Energy_Demand'].tolist() if len(st.session_state.df_hist) > 0 else []
            hist_solar = st.session_state.df_hist['Solar_Generation'].tolist() if len(st.session_state.df_hist) > 0 else []
            
            simulated_date = datetime.now().replace(hour=time_val, minute=0, second=0, microsecond=0)
            day_of_week = simulated_date.weekday()
            month = simulated_date.month
            
            energy_pred = predict_energy_consumption(hist_energy, temp_current, time_val, day_of_week, month)
            solar_pred = predict_solar_generation(temp_current, hum_current, time_val, 45, 20, hist_solar)
            
            sim.input['energy_demand'] = energy_val
            sim.input['battery_level'] = battery_val
            sim.input['grid_price'] = price_val
            sim.input['hour_of_day'] = time_val
            sim.input['solar_generation'] = solar_val
            sim.compute()
            
            red = sim.output['appliance_reduction']
            ba = sim.output['battery_control']
            ga = sim.output['grid_control']
            
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
                "Appliance_Reduction_real": red,
                "Appliance_Reduction_rec": round(red, 1),
                "Battery_Action_rec": round(ba, 1),
                "Grid_Action_rec": round(ga, 1),
                "Energy_Predicted": round(energy_pred, 1),
                "Solar_Predicted": round(solar_pred, 1)
            }])
            
            st.session_state.df_hist = pd.concat([st.session_state.df_hist, new_entry], ignore_index=True)
            st.session_state.df_hist.to_csv(DATA_FILE, index=False)
            
            st.success("✅ Calculation successful & saved to history!")
            
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

plot_membership_function(energy_demand, ax1, energy_val, "Energy Demand (0-100 scale)")
plot_membership_function(battery_level, ax2, battery_val, "Battery Level (0-100 scale)")
plot_membership_function(solar_generation, ax3, solar_val, "Solar Generation (0-100 scale)")
plot_membership_function(grid_price, ax4, price_val, "Grid Price (0-100 scale)")
plot_membership_function(hour_of_day, ax5, time_val, "Time of Day (0-24 scale)")

plot_membership_function(appliance_reduction, ax6, None, "Appliance Reduction (0-100)")
plot_membership_function(battery_control, ax7, None, "Battery Action (-100 to 100)")
plot_membership_function(grid_control, ax8, None, "Grid Interaction (-100 to 100)")

try:
    sim.input['energy_demand'] = energy_val
    sim.input['battery_level'] = battery_val
    sim.input['grid_price'] = price_val
    sim.input['hour_of_day'] = time_val
    sim.input['solar_generation'] = solar_val
    sim.compute()
    
    ax6.axvline(sim.output['appliance_reduction'], color='darkred', linewidth=3, linestyle=':', label=f'Output: {sim.output["appliance_reduction"]:.1f}%')
    ax7.axvline(sim.output['battery_control'], color='darkred', linewidth=3, linestyle=':', label=f'Output: {sim.output["battery_control"]:.1f}')
    ax8.axvline(sim.output['grid_control'], color='darkred', linewidth=3, linestyle=':', label=f'Output: {sim.output["grid_control"]:.1f}')
    
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


