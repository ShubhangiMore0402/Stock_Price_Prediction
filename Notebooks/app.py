import streamlit as st
import numpy as np
import joblib
from keras.models import load_model

# Load the trained LSTM model and scaler
MODEL_PATH = 'Notebooks/lstm_model.h5'
SCALER_PATH = 'Notebooks/scaler_X.pkl'

try:
    model = load_model(MODEL_PATH)
    scaler_X = joblib.load(SCALER_PATH)
    st.success("✅ Model and scaler loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")

# Streamlit UI
st.title("📈 Stock Price Prediction App")
st.write("Enter a stock symbol to predict its price:")

# User input
stock_symbol = st.text_input("Stock Symbol (e.g., AAPL, GOOGL, NFLX)", "")

if st.button("Predict Price"):
    if stock_symbol:
        st.write(f"Predicting price for {stock_symbol}...")

        # Replace this with actual fetching of stock data
        new_data = np.random.rand(1, 3)  # Dummy data, replace with real stock data
        new_data_scaled = scaler_X.transform(new_data)
        new_data_scaled = new_data_scaled.reshape((1, new_data_scaled.shape[0], new_data_scaled.shape[1]))

        # Predict
        prediction = model.predict(new_data_scaled)
        predicted_price = round(prediction[0][0], 2)

        st.success(f"🔮 Predicted Price for {stock_symbol}: **${predicted_price}**")
    else:
        st.warning("⚠️ Please enter a valid stock symbol.")