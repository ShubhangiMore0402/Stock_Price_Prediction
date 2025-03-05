import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import requests
import joblib

# Load dataset
data_path = 'Dataset/portfolio_data.csv'  # Update with your dataset path
df = pd.read_csv(data_path)

# Display the first few rows of the dataframe
print(df.head())

# Assume the target variable is 'NFLX'. Adjust if necessary.
target_column = 'NFLX'  # Replace with your actual target column name
X = df.drop([target_column, 'Date'], axis=1).values  # Features (drop target and date)
y = df[target_column].values  # Target variable

# Train-test split for Linear Regression and Random Forest
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)

# Metrics for Linear Regression
lin_mse = mean_squared_error(y_test, y_pred_lin)
lin_mae = mean_absolute_error(y_test, y_pred_lin)
lin_r2 = r2_score(y_test, y_pred_lin)
print(f"Linear Regression MSE: {lin_mse:.2f}, MAE: {lin_mae:.2f}, R²: {lin_r2:.2f}")

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Metrics for Random Forest
rf_mse = mean_squared_error(y_test, y_pred_rf)
rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_r2 = r2_score(y_test, y_pred_rf)
print(f"Random Forest MSE: {rf_mse:.2f}, MAE: {rf_mae:.2f}, R²: {rf_r2:.2f}")

# Scale the features and target variable
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Prepare data for LSTM
time_step = 60  # Set the time step
X_lstm, y_lstm = [], []

# Create sequences for LSTM
for i in range(time_step, len(X_scaled)):
    X_lstm.append(X_scaled[i-time_step:i])  # Last 'time_step' rows as features
    y_lstm.append(y_scaled[i])              # The corresponding target value

# Convert to numpy arrays
X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

# Train-test split for LSTM
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
model.add(Dropout(0.3))
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the LSTM model
model.fit(X_train_lstm, y_train_lstm, epochs=100, batch_size=32)

# Predictions and inverse scaling
y_pred_lstm_scaled = model.predict(X_test_lstm)
y_pred_lstm = scaler_y.inverse_transform(y_pred_lstm_scaled)  # Inverse scale the predictions
y_test_lstm_inv = scaler_y.inverse_transform(y_test_lstm.reshape(-1, 1))

# Metrics for LSTM
lstm_mse = mean_squared_error(y_test_lstm_inv, y_pred_lstm)
lstm_mae = mean_absolute_error(y_test_lstm_inv, y_pred_lstm)
lstm_r2 = r2_score(y_test_lstm_inv, y_pred_lstm)
print(f"LSTM MSE: {lstm_mse:.2f}, MAE: {lstm_mae:.2f}, R²: {lstm_r2:.2f}")

# Real-time prediction
# Replace with your actual latest prices
latest_amzn_price = 3200  # Example latest price for Amazon
latest_dpz_price = 150  # Example latest price for Domino's Pizza
latest_btc_price = 40000  # Example latest price for Bitcoin

# Prepare new data for prediction
new_data = np.array([[latest_amzn_price, latest_dpz_price, latest_btc_price]])  # Use actual latest prices
new_data_scaled = scaler_X.transform(new_data)  # Scale the new data

# Prepare new_data for LSTM (time step 60)
new_data_lstm = []
new_data_lstm.append(new_data_scaled)  # Adding the latest scaled data

# Convert to numpy array and reshape for LSTM
new_data_lstm = np.array(new_data_lstm)
new_data_lstm = new_data_lstm.reshape((new_data_lstm.shape[0], new_data_lstm.shape[1], new_data_lstm.shape[2]))

# Make the prediction
predicted_price_scaled = model.predict(new_data_lstm)
predicted_price = scaler_y.inverse_transform(predicted_price_scaled)  # Inverse scale the prediction

print(f"Predicted Price for NFLX: {predicted_price[0][0]:.2f}")

scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)

# Save the scaler
joblib.dump(scaler_X, 'Notebooks/scaler_X.pkl')
print("✅ Scaler saved successfully!")
model.save("Notebooks/lstm_model.h5")
print("Model saved as 'lstm_model.h5'")