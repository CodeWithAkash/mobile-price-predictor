
import streamlit as st
import joblib
import numpy as np

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("📱 Mobile Phone Price Predictor")

battery_power = st.slider("Battery Power (mAh)", 500, 2000)
blue = st.radio("Bluetooth", [0, 1])
clock_speed = st.slider("Clock Speed (GHz)", 0.5, 3.0, step=0.1)
dual_sim = st.radio("Dual SIM", [0, 1])
fc = st.slider("Front Camera (MP)", 0, 20)
four_g = st.radio("4G Support", [0, 1])
int_memory = st.slider("Internal Memory (GB)", 2, 128)
m_dep = st.slider("Mobile Depth (cm)", 0.1, 1.0, step=0.1)
mobile_wt = st.slider("Mobile Weight (g)", 80, 250)
n_cores = st.slider("Number of Cores", 1, 8)
pc = st.slider("Primary Camera (MP)", 0, 20)
px_height = st.slider("Pixel Height", 0, 2000)
px_width = st.slider("Pixel Width", 500, 2000)
ram = st.slider("RAM (MB)", 256, 4000)
sc_h = st.slider("Screen Height (cm)", 5, 20)
sc_w = st.slider("Screen Width (cm)", 2, 15)
talk_time = st.slider("Talk Time (hours)", 2, 20)
three_g = st.radio("3G Support", [0, 1])
touch_screen = st.radio("Touch Screen", [0, 1])
wifi = st.radio("WiFi", [0, 1])

if st.button("Predict"):
    features = np.array([[battery_power, blue, clock_speed, dual_sim, fc, four_g,
                          int_memory, m_dep, mobile_wt, n_cores, pc, px_height,
                          px_width, ram, sc_h, sc_w, talk_time, three_g,
                          touch_screen, wifi]])
    scaled_features = scaler.transform(features)
    result = model.predict(scaled_features)
    labels = ["Low Cost", "Medium Cost", "High Cost", "Very High Cost"]
    st.success(f"Predicted Price Range: {labels[result[0]]}")
