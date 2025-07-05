import pandas as pd
import numpy as np
import joblib
import streamlit as st
from io import StringIO

# -------- PAGE SETUP --------
st.set_page_config(page_title="Water Pollution Predictor", layout="centered")

# -------- WORKING FLOATING BUBBLES EFFECT --------
st.markdown("""
<style>
/* BUBBLE BACKGROUND CONTAINER */
.bubble-container {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  z-index: -10;
  overflow: hidden;
  pointer-events: none;
}

/* BUBBLE STYLE */
.bubble {
  position: absolute;
  bottom: -50px;
  width: 20px;
  height: 20px;
  background: rgba(255, 255, 255, 0.3);
  border-radius: 50%;
  animation: rise 15s infinite ease-in;
}

/* Different sizes & speeds */
.bubble:nth-child(1) { left: 10%; animation-duration: 14s; width: 15px; height: 15px; }
.bubble:nth-child(2) { left: 25%; animation-duration: 16s; width: 20px; height: 20px; }
.bubble:nth-child(3) { left: 40%; animation-duration: 18s; width: 25px; height: 25px; }
.bubble:nth-child(4) { left: 55%; animation-duration: 12s; width: 18px; height: 18px; }
.bubble:nth-child(5) { left: 70%; animation-duration: 20s; width: 22px; height: 22px; }
.bubble:nth-child(6) { left: 85%; animation-duration: 17s; width: 16px; height: 16px; }

@keyframes rise {
  0% {
    bottom: -60px;
    transform: translateX(0);
  }
  50% {
    transform: translateX(20px);
  }
  100% {
    bottom: 110%;
    transform: translateX(-20px);
  }
}
</style>

<div class="bubble-container">
  <div class="bubble"></div>
  <div class="bubble"></div>
  <div class="bubble"></div>
  <div class="bubble"></div>
  <div class="bubble"></div>
  <div class="bubble"></div>
</div>
""", unsafe_allow_html=True)

# -------- CUSTOM STYLING (RIPPLE + FONTS) --------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        background-image: url('https://i.ibb.co/1dQmYvH/ripple-bg.jpg');
        background-size: cover;
        background-attachment: fixed;
    }

    .main {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 2rem;
        border-radius: 12px;
    }

    h1, h2, h3 {
        color: #4682B4;
    }
    </style>
""", unsafe_allow_html=True)

# -------- TITLE & LOGO --------
col1, col2 = st.columns([1, 8])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/4005/4005901.png", width=60)
with col2:
    st.markdown("<h2>💧 Water Pollution Predictor</h2>", unsafe_allow_html=True)

# -------- LOAD MODEL --------
model = joblib.load("pollution_model.pkl")
model_cols = joblib.load("model_columns.pkl")

# -------- TABS --------
tabs = st.tabs(["🔍 Prediction", "⚙️ How It Works", "📩 Contact"])

# -------- TAB 1: Prediction --------
with tabs[0]:
    st.markdown("Use this tool to predict water pollutant levels for any station and year.")

    with st.form("pollution_form"):
        year_input = st.number_input("📅 Enter Year", min_value=2000, max_value=2100, value=2022)
        station_id = st.text_input("🛰️ Enter Station ID", value='1')
        submit = st.form_submit_button("🔍 Predict")

    if submit:
        if not station_id:
            st.warning('⚠️ Please enter the Station ID')
        else:
            with st.spinner("🔄 Predicting pollutant levels..."):
                input_df = pd.DataFrame({'year': [year_input], 'id': [station_id]})
                input_encoded = pd.get_dummies(input_df, columns=['id'])

                for col in model_cols:
                    if col not in input_encoded.columns:
                        input_encoded[col] = 0
                input_encoded = input_encoded[model_cols]

                predicted_pollutants = model.predict(input_encoded)[0]
                pollutants = ['NH4', 'BSK5', 'Suspended', 'O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']
                result_df = pd.DataFrame({
                    "Pollutant": pollutants,
                    "Value": [round(val, 2) for val in predicted_pollutants]
                })

            st.success(f"✅ Predicted pollutant levels for Station **{station_id}** in **{year_input}**")
            st.dataframe(result_df, use_container_width=True)

            st.markdown("### 📊 Visualization")
            st.bar_chart(result_df.set_index("Pollutant"))

            csv_buffer = StringIO()
            result_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="⬇️ Download Results as CSV",
                data=csv_buffer.getvalue(),
                file_name=f"pollution_prediction_{station_id}_{year_input}.csv",
                mime='text/csv'
            )

# -------- TAB 2: How It Works --------
with tabs[1]:
    st.markdown("### ⚙️ How the Model Works")
    st.markdown("""
    - This tool uses a **machine learning regression model** trained on historical water pollutant data.
    - Input parameters include:
        - `Year`: Represents the target year
        - `Station ID`: Identifies the water monitoring station
    - The model predicts levels of pollutants such as **NH4**, **NO2**, **O2**, and others.
    - One-hot encoding is applied to station IDs, and the model outputs pollutant levels in mg/L or appropriate units.
    """)

# -------- TAB 3: Contact --------
with tabs[2]:
    st.markdown("### 📩 Contact")
    st.markdown("""
    Created by **Nandish Patil**

    🔗 [LinkedIn](https://www.linkedin.com/in/nandish22/)  
    📧 [nandishpatil14@gmail.com](mailto:nandishpatil14@gmail.com)

    Feel free to connect or reach out for collaboration!
    """)
