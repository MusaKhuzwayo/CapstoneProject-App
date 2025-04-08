pip install streamlit   install streaMLIT
pip install streamlit-lottie
pip install scikit-learn
pip install prophet
pip install fuzzywuzzy
pip install python-Levenshtein
pip install plotly 
pip install pandas
pip install streaMLIT


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image
import requests
from io import BytesIO
import time
import json
from streamlit_lottie import st_lottie
from sklearn.linear_model import LinearRegression  # Import Linear Regression
from prophet import Prophet  # Import Prophet
from fuzzywuzzy import fuzz  # Import fuzzywuzzy for string matching

# Set page config
st.set_page_config(page_title="Veggie Price Predictor", layout="wide", page_icon="🥦")


# --- Load animation ---
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Sample Lottie animation with fallback
lottie_url = "https://assets2.lottiefiles.com/packages/lf20_bu3xvx.json"
lottie_veggie = load_lottieurl(lottie_url)

if lottie_veggie is None:
    st.warning("Lottie animation loading failed. Using fallback (if available).")
    try:
        with open("fallback_animation.json", "r") as f:
            lottie_veggie = json.load(f)
    except FileNotFoundError:
        st.warning("Fallback animation not found.")
        lottie_veggie = None  # Ensure lottie_veggie is None if fallback fails


# --- Mock data and images ---
veggie_images = {
    "Tomato": "https://source.unsplash.com/600x400/?tomato",
    "Spinach": "https://source.unsplash.com/600x400/?spinach",
    "Carrot": "https://source.unsplash.com/600x400/?carrot",
    "Broccoli": "https://source.unsplash.com/600x400/?broccoli",
    "Potato": "https://source.unsplash.com/600x400/?potato",
}

data = pd.DataFrame({
    "Vegetable": ["Tomato", "Spinach", "Carrot", "Broccoli", "Potato"],
    "Avg_Price": [12.5, 8.0, 10.2, 14.3, 7.5],
    "Predicted_Date": ["2025-04-08"] * 5,
})

# --- Prediction models ---
def predict_price(model_name, selected_veg):
    """Simulates price prediction using different models."""
    # In a real-world scenario, replace with actual model training and prediction.

    result = data[data["Vegetable"] == selected_veg].iloc[0].copy()
    # Add some variation for different models
    if model_name == "Linear Regression":
        result['Avg_Price'] *= 1.05
    elif model_name == "Decision Tree":
        result['Avg_Price'] *= 1.10
    # ... add more variations for other models ...
    return result

    # --- Fuzzy Matching Function ---
def fuzzy_match_vegetable(user_input, valid_vegetables):
    """Matches user input to the closest valid vegetable name."""
    best_match = None
    highest_score = 0

    for vegetable in valid_vegetables:
        score = fuzz.ratio(user_input.lower(), vegetable.lower())  # Case-insensitive matching
        if score > highest_score:
            highest_score = score
            best_match = vegetable

    # Set a threshold for matching (e.g., 80%)
    if highest_score >= 80:
        return best_match
    else:
        return None  # No close match found

# --- App layout ---
st.markdown("""
    <style>
        .main {
            background-color: #0f0f0f;
            color: white;
        }
        .block-container {
            padding-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🥦 Veggie Price Predictor")
st.subheader("Accurate, Creative and Easy-to-Use Pricing Tool")

# --- Sidebar ---
with st.sidebar:
    if lottie_veggie:
        st_lottie(lottie_veggie, height=200)
    st.markdown("**Select a vegetable and a model to predict pricing trends.**")

    # --- User Input for Vegetable ---
    user_input = st.text_input("Enter Vegetable Name:", "")

    # --- Fuzzy Matching ---
    valid_vegetables = data['Vegetable'].unique()  # Get valid vegetable names
    matched_vegetable = fuzzy_match_vegetable(user_input, valid_vegetables)

    # --- Selectbox with Matched Vegetable ---
    if matched_vegetable:
        selected_veg = st.selectbox("Choose a Vegetable", valid_vegetables, index=valid_vegetables.tolist().index(matched_vegetable))
    else:
        selected_veg = st.selectbox("Choose a Vegetable", valid_vegetables)
    model = st.selectbox("Select Prediction Model", [
        "Linear Regression", "Decision Tree", "Random Forest", "SVR", "XGBoost", "Prophet"
    ])
    st.markdown("---")
    if st.button("Predict"):
        with st.spinner("Crunching the numbers..."):
            time.sleep(2)  # simulate prediction

            # --- Error handling ---
            if selected_veg not in veggie_images:
                st.error("Invalid vegetable selection.")

            # --- Prediction ---
            result = predict_price(model, selected_veg) 

            # --- Layout ---
            col1, col2 = st.columns([1, 1])

            with col1:
                st.image(veggie_images[selected_veg], caption=selected_veg, use_column_width=True)

            with col2:
                st.markdown(f"### 🥗 Prediction Results for {selected_veg}")
                st.markdown(f"**Model Used:** {model}")
                st.markdown(f"**Predicted Avg Price:** R{result['Avg_Price']:.2f}")
                st.markdown(f"**Expected Date:** {result['Predicted_Date']}")

                # --- Price Trend Chart ---
                chart_data = pd.DataFrame({
                    "Day": ["Yesterday", "Today", "Tomorrow"],
                    "Price": [result["Avg_Price"] * 0.95, result["Avg_Price"], result["Avg_Price"] * 1.05]
                })
                fig = px.line(chart_data, x="Day", y="Price", title=f"{selected_veg} Price Trend")
                st.plotly_chart(fig, use_container_width=True)

                st.success("Prediction complete!") 

# --- Footer ---
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | © 2025 Your Name")
#Removed the extra code that was causing the IndentationError because it was outside the main execution flow or any function.
