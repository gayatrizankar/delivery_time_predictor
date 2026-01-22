import streamlit as st
import pickle
import numpy as np

# Page config
st.set_page_config(
    page_title="Delivery Time Predictor 🚴",
    page_icon="🚴",
    layout="wide"
)

# Load trained model
model = pickle.load(open("delivery_time_model.pkl", "rb"))

# Title
st.markdown("""
# 🚴 Delivery Time Prediction App
Predict how long your order will take to arrive using Machine Learning.
""")

st.markdown("---")

# Sidebar Inputs
st.sidebar.header("📦 Order Details")

distance_km = st.sidebar.number_input("Distance (km)", 0.1, 50.0, 5.0)
traffic_level = st.sidebar.selectbox("Traffic Level", [0, 1, 2, 3], format_func=lambda x: {
    0: "Low",
    1: "Medium",
    2: "High",
    3: "Very High"
}[x])

weather = st.sidebar.selectbox("Weather", [0, 1, 2], format_func=lambda x: {
    0: "Clear",
    1: "Rain",
    2: "Storm"
}[x])

prep_time = st.sidebar.number_input("Preparation Time (minutes)", 1, 120, 15)

# Show input summary
st.subheader("📋 Order Summary")

col1, col2 = st.columns(2)

with col1:
    st.write(f"**Distance:** {distance_km} km")
    st.write(f"**Traffic:** {['Low','Medium','High','Very High'][traffic_level]}")

with col2:
    st.write(f"**Weather:** {['Clear','Rain','Storm'][weather]}")
    st.write(f"**Preparation Time:** {prep_time} min")

# Predict Button
if st.button("⏱️ Predict Delivery Time"):
    input_data = np.array([[distance_km, traffic_level, weather, prep_time]])
    prediction = model.predict(input_data)

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    predicted_time = round(prediction[0], 1)

    st.success(f"🚚 Estimated Delivery Time: **{predicted_time} minutes**")

    # Simple interpretation
    if predicted_time <= 30:
        st.balloons()
        st.info("⚡ Super fast delivery expected!")
    elif predicted_time <= 60:
        st.info("🙂 Normal delivery time.")
    else:
        st.warning("⏳ Delivery may take longer due to distance or traffic.")
