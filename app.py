import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os
import pickle

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="📊 Stock Market Portfolio Dashboard",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: white;
        font-family: 'Segoe UI';
    }
    h1, h2, h3, h4, h5 {
        color: #00FFFF;
    }
    .stButton>button {
        background-color: #00FFFF;
        color: black;
        font-weight: bold;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.header("⚙️ Settings")
st.sidebar.write("Upload your stock CSV or use default data")
uploaded_file = st.sidebar.file_uploader("📁 CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("stock_data.csv")

# --- DATA CLEANING ---
df = df[pd.to_numeric(df["Open"], errors="coerce").notnull()]
df = df.reset_index(drop=True)

# --- DISPLAY DATA ---
st.title("💹 Stock Market Movement Prediction Dashboard")
st.write("Predict if the stock price will go **UP 📈 or DOWN 📉** tomorrow based on technical indicators")
st.dataframe(df.tail())

# --- FEATURES & TARGET ---
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Return', 'RSI', 'MACD', 'EMA_10', 'EMA_30']
target = 'Target_Cls'

X = df[features].apply(pd.to_numeric, errors='coerce')
y = df[target]
X = X.dropna()
y = y.loc[X.index]

# --- TRAIN OR LOAD MODEL ---
model_path = "stock_market.pkl"

if os.path.exists(model_path):
    st.success("✅ Loaded pre-trained model from stock_market.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
else:
    st.info("⚙️ Training Random Forest model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    st.success("✅ Model trained and saved as stock_market.pkl")

# --- EVALUATE MODEL ---
st.subheader("📊 Model Performance")
if 'X_test' not in locals():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
st.metric("Accuracy", f"{accuracy*100:.2f}%")

with st.expander("Show Classification Report"):
    report = classification_report(y_test, y_pred, output_dict=True)
    st.json(report)

# --- FEATURE IMPORTANCE ---
st.subheader("🌟 Feature Importance")
fig, ax = plt.subplots()
feature_imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
feature_imp.plot(kind="barh", color="#00FFFF", ax=ax)
st.pyplot(fig)

# --- TOMORROW'S PREDICTION ---
latest_data = X.iloc[[-1]]
tomorrow_pred = model.predict(latest_data)[0]
prob = model.predict_proba(latest_data)[0]

movement = "UP 📈" if tomorrow_pred == 1 else "DOWN 📉"
color = "#00FF00" if tomorrow_pred == 1 else "#FF4B4B"

st.markdown("---")
st.subheader("🕒 Tomorrow's Stock Movement Prediction")
st.markdown(f"### The stock will likely go **<span style='color:{color}'>{movement}</span>**", unsafe_allow_html=True)
st.progress(float(prob[1]) if tomorrow_pred == 1 else float(prob[0]))
st.write(f"**Down (0):** {prob[0]*100:.2f}%  |  **Up (1):** {prob[1]*100:.2f}%")

# --- USER INPUT FOR NEW PREDICTION ---
st.markdown("---")
st.subheader("📥 Try Your Own Stock Values")

col1, col2, col3 = st.columns(3)
with col1:
    open_ = st.number_input("Open", value=2850.0)
    high = st.number_input("High", value=2900.0)
    low = st.number_input("Low", value=2840.0)
with col2:
    close = st.number_input("Close", value=2890.0)
    volume = st.number_input("Volume", value=1200000)
    ret = st.number_input("Return", value=0.015)
with col3:
    rsi = st.number_input("RSI", value=65.0)
    macd = st.number_input("MACD", value=1.2)
    ema10 = st.number_input("EMA_10", value=2875.0)
    ema30 = st.number_input("EMA_30", value=2800.0)

if st.button("🔮 Predict Movement"):
    new_data = pd.DataFrame([{
        "Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume,
        "Return": ret, "RSI": rsi, "MACD": macd, "EMA_10": ema10, "EMA_30": ema30
    }])
    new_pred = model.predict(new_data)[0]
    new_prob = model.predict_proba(new_data)[0]
    move = "UP 📈" if new_pred == 1 else "DOWN 📉"
    color = "#00FF00" if new_pred == 1 else "#FF4B4B"
    st.markdown(f"### Predicted movement: <span style='color:{color}'>{move}</span>", unsafe_allow_html=True)
    st.write(f"**Down (0):** {new_prob[0]*100:.2f}% | **Up (1):** {new_prob[1]*100:.2f}%")

# --- MINI KPI DASHBOARD ---
st.markdown("---")
st.subheader("📊 Portfolio KPIs")

# Simple KPI for this single stock example
expected_profit = prob[1] * latest_data['Close'].iloc[0]
risk_indicator = prob[0] * 100
st.metric("Expected Profit ($)", f"{expected_profit:.2f}")
st.metric("Portfolio Risk (%)", f"{risk_indicator:.2f}%")
st.markdown(f"Movement: <span style='color:{color}'>{movement}</span>", unsafe_allow_html=True)

st.markdown("---")
st.caption("🧠 Created by Raj Kumar | Machine Learning Project | Streamlit Dashboard")
