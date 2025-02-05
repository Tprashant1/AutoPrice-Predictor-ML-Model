import pandas as pd
import numpy as np
import os
import pickle as pk
import streamlit as st

# Load the trained model
# model = pk.load(open('model.pkl', 'rb'))

model_path = os.path.join(os.path.dirname(__file__), "model.pkl")

# Load model safely
with open(model_path, "rb") as file:
    model = pk.load(file)

# Set Page Config
st.set_page_config(page_title="AutoPrice Predictor", page_icon="🚗", layout="wide")

# Apply Custom CSS for Styling
st.markdown(
    """
    <style>
        body {background-color: #F5F7FA;}
        .stButton>button {background-color: #4CAF50; color: white; font-size: 18px; border-radius: 10px; padding: 10px 20px;}
        .stMarkdown {font-size: 20px; color: #2E86C1; font-weight: bold;}
        .stSelectbox, .stSlider, .stButton {margin-bottom: 20px !important;}
    </style>
    """,
    unsafe_allow_html=True
)

# App Header
st.markdown("<h1 style='text-align: center; color: #2E86C1; font-size: 48px;'>🚗 AutoPrice Predictor ML Model</h1>", unsafe_allow_html=True)

# Load Data
cars_data = pd.read_csv('Cardetails.csv')

def get_brand_name(car_name):
    return car_name.split(' ')[0].strip()

cars_data['name'] = cars_data['name'].apply(get_brand_name)

# User Inputs
col1, col2 = st.columns(2, gap="large")

with col1:
    name = st.selectbox('🚘 Select Car Brand', cars_data['name'].unique())
    year = st.slider('📅 Car Manufactured Year', 1994, 2024, step=1)
    km_driven = st.slider('🛣️ No of kms Driven', 11, 200000, step=500)
    fuel = st.selectbox('⛽ Fuel Type', cars_data['fuel'].unique())
    seller_type = st.selectbox('🧑‍💼 Seller Type', cars_data['seller_type'].unique())

with col2:
    transmission = st.selectbox('⚙️ Transmission Type', cars_data['transmission'].unique())
    owner = st.selectbox('👤 Ownership Type', cars_data['owner'].unique())
    mileage = st.slider('⛽ Car Mileage (km/l)', 10, 40, step=1)
    engine = st.slider('🚀 Engine CC', 700, 5000, step=100)
    max_power = st.slider('⚡ Max Power (BHP)', 0, 200, step=10)
    seats = st.slider('🛋️ No of Seats', 5, 10, step=1)

# Prediction Button
if st.button("🔍 Predict Price"):
    input_data_model = pd.DataFrame(
        [[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
        columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']
    )
    
    # Encode categorical values
    input_data_model['owner'].replace(['First Owner', 'Second Owner', 'Third Owner',
       'Fourth & Above Owner', 'Test Drive Car'],
       [1, 2, 3, 4, 5], inplace=True)
    input_data_model['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4], inplace=True)
    input_data_model['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3], inplace=True)
    input_data_model['transmission'].replace(['Manual', 'Automatic'], [1, 2], inplace=True)
    input_data_model['name'].replace(
        ['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
         'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
         'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
         'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
         'Ambassador', 'Ashok', 'Isuzu', 'Opel'],
        range(1, 32), inplace=True)
    
    # Make Prediction
    car_price = model.predict(input_data_model)
    
    # Display Result
    st.markdown(f"<h2 style='text-align: center; color: green;'>💰 Estimated Car Price: ₹{car_price[0]:,.2f}</h2>", unsafe_allow_html=True)
