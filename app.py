import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model

# ── Load assets ──────────────────────────────────────────────────────────────
scaler = joblib.load('tsla_scaler.pkl')
model  = load_model('tsla_lstm_best.keras')
df     = pd.read_csv('TSLA.csv', parse_dates=['Date'], index_col='Date')
df.sort_index(inplace=True)
df.ffill(inplace=True)

LOOKBACK = 60

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
horizon = st.sidebar.selectbox("Prediction Horizon", [1, 5, 10], format_func=lambda x: f"{x}-day")
n_future = st.sidebar.slider("Days to forecast ahead", 1, 30, 10)

st.title("🚗 Tesla Stock Price Predictor")
st.markdown("Using LSTM deep learning model trained on TSLA historical data (2010–2020)")

# ── Actual price chart ────────────────────────────────────────────────────────
st.subheader("📈 Historical Closing Price")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df.index, df['Adj Close'], color='#1f77b4', linewidth=1)
ax.set_ylabel("Price (USD)")
st.pyplot(fig)

# ── Predict on test set ───────────────────────────────────────────────────────
scaled = scaler.transform(df[['Adj Close']])
split  = int(len(scaled) * 0.8)
test   = scaled[split:]

X_test, y_test = [], []
for i in range(LOOKBACK, len(test) - horizon + 1):
    X_test.append(test[i - LOOKBACK:i, 0])
    y_test.append(test[i + horizon - 1, 0])

X_test  = np.array(X_test)[..., np.newaxis]
y_test  = np.array(y_test)

preds   = scaler.inverse_transform(model.predict(X_test))
actuals = scaler.inverse_transform(y_test.reshape(-1, 1))

# ── Actual vs Predicted ───────────────────────────────────────────────────────
st.subheader(f"🔮 Actual vs Predicted — {horizon}-day horizon")
fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.plot(actuals, label='Actual',    color='#1f77b4')
ax2.plot(preds,   label='Predicted', color='#d62728', linestyle='--')
ax2.legend(); ax2.set_ylabel("Price (USD)")
st.pyplot(fig2)

# ── Metrics ───────────────────────────────────────────────────────────────────
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
rmse = np.sqrt(mean_squared_error(actuals, preds))
mae  = mean_absolute_error(actuals, preds)
r2   = r2_score(actuals, preds)

st.subheader("📊 Model Performance")
col1, col2, col3 = st.columns(3)
col1.metric("RMSE", f"${rmse:.2f}")
col2.metric("MAE",  f"${mae:.2f}")
col3.metric("R²",   f"{r2:.4f}")

# ── Forecast next N days ──────────────────────────────────────────────────────
st.subheader(f"📅 Forecast Next {n_future} Days")
last_window = scaled[-LOOKBACK:].reshape(1, LOOKBACK, 1)
future_preds = []

for _ in range(n_future):
    pred = model.predict(last_window, verbose=0)[0, 0]
    future_preds.append(pred)
    last_window = np.append(last_window[:, 1:, :],
                            [[[pred]]], axis=1)

future_prices = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
future_dates  = pd.date_range(df.index[-1], periods=n_future+1, freq='B')[1:]

fig3, ax3 = plt.subplots(figsize=(10, 4))
ax3.plot(df.index[-60:], df['Adj Close'].iloc[-60:], color='#1f77b4', label='Historical')
ax3.plot(future_dates,   future_prices,              color='#d62728',
         linestyle='--', marker='o', markersize=4,   label='Forecast')
ax3.legend(); ax3.set_ylabel("Price (USD)")
st.pyplot(fig3)

st.caption("⚠️ This is for educational purposes only — not financial advice.")