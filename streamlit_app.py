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
from sklearn.linear_model import LinearRegression
from prophet import Prophet
from fuzzywuzzy import fuzz

# Set page config
st.set_page_config(page_title="Veggie Price Predictor", layout="wide", page_icon="🥦")

# --- Load animation ---
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url = "https://assets2.lottiefiles.com/packages/lf20_bu3xvx.json"
lottie_veggie = load_lottieurl(lottie_url)

# --- Handling Lottie Animation Loading ---
if lottie_veggie is None:
    st.warning("Lottie animation loading failed. Using fallback (if available).")
    try:
        with open("assets/fallback_animation.json", "r") as f:
            lottie_veggie = json.load(f)
    except FileNotFoundError:
        st.warning("Fallback animation not found.")
        lottie_veggie = None

# --- Mock data and images ---
veggie_images = {
    "Tomato": "tomatoes-1280859_1280.jpg",
    "Onion": "photo-1620574387735-3624d75b2dbc.jpeg",
    "Carrot": "pexels-mali-65174.jpg",
    "Broccoli": "https://source.unsplash.com/600x400/?broccoli",
    "Potato": "premium_photo-1675365779531-031dfdcdf947.jpeg",
    "Brinjal": "pexels-freestockpro-321551.jpg",
    "Garlic": "pexels-minan1398-1638522.jpg",
    "Peas": "pexels-pixabay-255469.jpg",
    "Methi": "36ac5dc3ddcc866d481bd585c277c236.jpg",
    "Green Chilli": "photo-1576763595295-c0371a32af78.jpeg",
    "Elephant Yam": "images.jpeg",
}

data = pd.DataFrame({
    "Vegetable": list(veggie_images.keys()),
    "Avg_Price": [12.5, 9.0, 10.2, 14.3, 7.5, 8.2, 15.0, 11.0, 7.8, 6.5, 13.2],
    "Predicted_Date": ["2025-04-08"] * len(veggie_images),
})

# --- Prediction models ---
def predict_price(model_name, selected_veg):
    result = data[data["Vegetable"] == selected_veg].iloc[0].copy()
    if model_name == "Linear Regression":
        result['Avg_Price'] *= 1.05
    elif model_name == "Decision Tree":
        result['Avg_Price'] *= 1.10
    return result

# --- Fuzzy Matching Function ---
def fuzzy_match_vegetable(user_input, valid_vegetables):
    best_match = None
    highest_score = 0
    for vegetable in valid_vegetables:
        score = fuzz.ratio(user_input.lower(), vegetable.lower())
        if score > highest_score:
            highest_score = score
            best_match = vegetable
    if highest_score >= 80:
        return best_match
    else:
        return None

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

    user_input = st.text_input("Enter Vegetable Name:", "")
    valid_vegetables = data['Vegetable'].unique()
    matched_vegetable = fuzzy_match_vegetable(user_input, valid_vegetables)

    if matched_vegetable:
        selected_veg = st.selectbox("Choose a Vegetable", valid_vegetables, index=valid_vegetables.tolist().index(matched_vegetable))
    else:
        selected_veg = st.selectbox("Choose a Vegetable", valid_vegetables)
    model = st.selectbox("Select Prediction Model", [
        "Linear Regression", "Decision Tree", "Random Forest", "SVR", "XGBoost", "Prophet"
    ])
    st.markdown("---")

# --- Main Content Area ---
if st.button("Predict"):
    with st.spinner("Crunching the numbers..."):
        time.sleep(2)

        if selected_veg not in veggie_images:
            st.error("Invalid vegetable selection.")
            return

        result = predict_price(model, selected_veg)

        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            st.image("assets/" + veggie_images[selected_veg], caption=selected_veg, use_container_width=True)

        with col2:
            st.markdown(f"### 🥗 Prediction Results for {selected_veg}")
            st.markdown(f"**Model Used:** {model}")
            st.markdown(f"**Predicted Avg Price:** R{result['Avg_Price']:.2f}")
            st.markdown(f"**Expected Date:** {result['Predicted_Date']}")

        with col3:
            chart_data = pd.DataFrame({
                "Day": ["Yesterday", "Today", "Tomorrow"],
                "Price": [result["Avg_Price"] * 0.95, result["Avg_Price"], result["Avg_Price"] * 1.05]
            })
            fig = px.bar(chart_data, x="Day", y="Price", title=f"{selected_veg} Price Trend")
            st.plotly_chart(fig, use_container_width=True)

        st.success("Prediction complete!")

# --- Footer ---
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | © 2025 Your Name")
