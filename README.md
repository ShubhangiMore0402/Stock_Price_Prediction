# 📈 Stock Price Prediction using Machine Learning & LSTM

This project predicts stock prices using **Linear Regression, Random Forest, and LSTM (Long Short-Term Memory Networks)**.  
It fetches stock data, processes it, and makes predictions through a **Streamlit web app**.

---

## 🚀 **Technologies Used**
- **Python** (Main Programming Language)
- **Pandas, NumPy** (Data Processing)
- **Scikit-learn** (Machine Learning Models)
- **TensorFlow/Keras** (LSTM Model)
- **Streamlit** (Web Application)
- **Joblib** (Saving and Loading Scalers & Models)

---

## 📊 **Model Performance**
| Model              | MSE   | MAE   | R²  |
|-------------------|------|------|----|
| **Linear Regression** | 502.68 | 16.75 | 0.96 |
| **Random Forest**  | 78.73 | 4.70  | 0.99 |
| **LSTM**           | 170.90 | 10.60 | 0.98 |

---

## 🛠️ **Project Setup**
### **1️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **2️⃣ Run the Streamlit App**
```bash
streamlit run app.py
```
This will start a local server where you can enter a stock symbol and get predictions.

---

## 🎯 **Features**
✅ **Real-time stock price fetching**  
✅ **Multiple models for prediction** (Linear Regression, Random Forest, LSTM)  
✅ **User-friendly web interface** using Streamlit  
✅ **Saved & Loaded trained models** for efficiency  

---

## 📌 **Future Enhancements**
- 📊 **Improve LSTM Accuracy** with hyperparameter tuning  
- 💡 **Add More Stock Indicators** for better predictions  
- 🌍 **Deploy on Cloud (AWS/GCP)** for public access  

---

## 🏆 **License**
This project is open-source and available under the [LICENSE](License)

