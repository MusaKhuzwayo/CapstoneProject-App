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
lottie_url = "https://assets2.lottiefiles.com/packages/lf20_bu3xvx.json"  # Replace with correct URL
lottie_veggie = load_lottieurl(lottie_url)

if lottie_veggie is None:
    st.warning("Lottie animation loading failed. Using fallback (if available).")
    try:
        with open("assets/fallback_animation.json", "r") as f:  # Update path if necessary
            lottie_veggie = json.load(f)
    except FileNotFoundError:
        st.warning("Fallback animation not found.")
        lottie_veggie = None  # Ensure lottie_veggie is None if fallback fails


# --- Mock data and images ---
veggie_images = {
    "Tomato": "assets/tomatoes-1280859_1280.jpg",  
    "Onion": "assets/photo-1620574387735-3624d75b2dbc.jpeg",  
    "Carrot": "assets/pexels-mali-65174.jpg",  
    "Broccoli": "https://source.unsplash.com/600x400/?broccoli",
    "Potato": "assets/premium_photo-1675365779531-031dfdcdf947.jpeg",  
    "Brinjal": "assets/pexels-freestockpro-321551.jpg",  
    "Garlic": "assets/pexels-minan1398-1638522.jpg",  
    "Peas": "assets/pexels-pixabay-255469.jpg",  
    "Methi": "assets/36ac5dc3ddcc866d481bd585c277c236.jpg",
    "Green Chilli": "assets/photo-1576763595295-c0371a32af78.jpeg",  
    "Elephant Yam": "assets/images.jpeg",
}

data = pd.DataFrame({
    "Vegetable": ["Tomato", "Onion", "Carrot", "Broccoli", "Potato", "Brinjal", "Garlic", "Peas", "Methi", "Green Chilli", "Elephant Yam"],
    "Avg_Price": [12.5, 9.0, 10.2, 14.3, 7.5, 8.2, 15.0, 11.0, 7.8, 6.5, 13.2],
    "Predicted_Date": ["2025-04-08"] * 11,  # Adjust length to match the number of vegetables
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
    # ... (Lottie animation and other sidebar elements)

    # --- Search Options ---
    st.markdown("### Search Options")
    search_term = st.text_input("Search by Vegetable Name:")
    start_date = st.date_input("Start Date:")
    end_date = st.date_input("End Date:")
    min_price = st.number_input("Minimum Price:", value=0.0)
    max_price = st.number_input("Maximum Price:", value=100.0)

    # --- Filter Data ---
    filtered_data = data.copy()  # Create a copy of the original data

    if search_term:
        filtered_data = filtered_data[filtered_data["Vegetable"].str.contains(search_term, case=False)]
    
    # Filter by date (assuming your data has a 'Date' column)
    # Replace 'Date' with the actual date column name in your DataFrame
    filtered_data['Predicted_Date'] = pd.to_datetime(filtered_data['Predicted_Date'])  # Convert to datetime
    filtered_data = filtered_data[(filtered_data['Predicted_Date'] >= pd.Timestamp(start_date)) & 
                                     (filtered_data['Predicted_Date'] <= pd.Timestamp(end_date))]

    filtered_data = filtered_data[(filtered_data["Avg_Price"] >= min_price) & (filtered_data["Avg_Price"] <= max_price)]

    # --- Display Filtered Data ---
    st.markdown("### Filtered Results")
    st.dataframe(filtered_data)

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
    
    # --- Main Content Area ---
    if st.button("Predict"):
        with st.spinner("Crunching the numbers..."):
             time.sleep(2)  # simulate prediction

        # --- Error handling ---
        if selected_veg not in veggie_images:
            st.error("Invalid vegetable selection.")

        # --- Prediction ---
        result = predict_price(model, selected_veg)

        # --- Layout ---
        col1, col2, col3 = st.columns([1, 2, 1])  # Create 3 columns

        with col1:  # Left bar for image
            st.image(veggie_images[selected_veg], caption=selected_veg, use_container_width=True)

        with col2:  # Middle column for predictions
            st.markdown(f"### 🥗 Prediction Results for {selected_veg}")
            st.markdown(f"**Model Used:** {model}")
            st.markdown(f"**Predicted Avg Price:** R{result['Avg_Price']:.2f}")
            st.markdown(f"**Expected Date:** {result['Predicted_Date']}")

        with col3:  # Right bar for chart
            chart_data = pd.DataFrame({
                "Day": ["Yesterday", "Today", "Tomorrow"],
                "Price": [result["Avg_Price"] * 0.95, result["Avg_Price"], result["Avg_Price"] * 1.05]
            })
            fig = px.bar(chart_data, x="Day", y="Price", title=f"{selected_veg} Price Trend")  # Changed to bar chart
            st.plotly_chart(fig, use_container_width=True)

        st.success("Prediction complete!")

# --- Footer ---
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | © 2025 Musa Khuzwayo")
#Removed the extra code that was causing the IndentationError because it was outside the main execution flow or any function.
