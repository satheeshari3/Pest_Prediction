# 🌾 Pest Prediction Using LSTM (Deep Learning)

This project predicts **weather conditions** (Max & Min Temperature) and then uses them to forecast **pest/disease outbreaks** based on agricultural data using **LSTM (Long Short-Term Memory)** models in TensorFlow/Keras.

---

## 📌 Project Goals

1. **Predict Max & Min Temperature** based on previous weather data.
2. **Predict Pest/Disease outbreaks** using predicted temperature and other environmental features.
3. Help farmers and agricultural scientists prepare early for pest attacks.

---

## 🛠️ Technologies Used

- Python 🐍
- TensorFlow / Keras 🤖
- NumPy / Pandas 📊
- scikit-learn 🔍
- LSTM (Deep Learning model)

---

## 📂 Dataset

The dataset file is named `cropdata.csv`. It contains:
- Weather data (temperature, humidity, rainfall, etc.)
- Year & week of observation
- Location
- Pest/Disease labels

> Features used:
- Observation Year
- Standard Week
- RH1(%), RH2(%)
- RF(mm), WS(kmph), SSH(hrs), EVP(mm)
- Location (encoded)
- Pest/Disease (encoded)

---

## 🔄 Project Workflow

1. **Read dataset** and clean column names.
2. **Encode categorical features** (Location, Pest/Disease).
3. **Scale data** using MinMaxScaler for better LSTM performance.
4. **Create sequences** for LSTM input using time steps.
5. **Train LSTM model** to predict MaxT and MinT.
6. **Use those predictions** to train another LSTM to predict pest/disease.
7. **Predict pest based on a user-given location.**

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/pest-prediction-lstm.git
cd pest-prediction-lstm
