import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Pricing Dashboard", layout="wide")

# -----------------------------
# MODERN UI CSS
# -----------------------------
st.markdown("""
<style>

/* Background */
.stApp {
    background-color: #0A0F1C;
    color: #E5E7EB;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #020617;
}

/* Main container */
.block-container {
    padding-top: 1.5rem;
    max-width: 1200px;
}

/* Title */
h1 {
    font-size: 30px !important;
    font-weight: 600;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.04);
    padding: 20px;
    border-radius: 14px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.08);
}

/* Metrics */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03);
    padding: 15px;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.05);
}

/* Button */
div.stButton > button {
    background: #2563EB;
    color: white !important;
    border-radius: 8px;
    height: 42px;
    font-weight: 600;
    border: none;
    transition: 0.3s;
}

div.stButton > button:hover {
    background: #1D4ED8;
    transform: translateY(-1px);
}

/* Inputs */
input {
    background-color: #020617 !important;
    color: #E5E7EB !important;
}

/* Divider */
hr {
    border-color: rgba(255,255,255,0.08);
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("best_model.pkl")
feature_columns = model.feature_names_in_

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("Pricing Controls")
price = st.sidebar.number_input("Product Price", value=100.0)

run = st.sidebar.button("Run Analysis")

# -----------------------------
# HEADER
# -----------------------------
st.title("Dynamic Pricing Dashboard")
st.caption("Machine learning powered revenue optimization")

st.markdown("---")

# -----------------------------
# FUNCTIONS
# -----------------------------
def predict_sales(price):
    df = pd.DataFrame([{"price": price}])
    df = df.reindex(columns=feature_columns, fill_value=0)
    return max(0, model.predict(df)[0])

def optimize_price(price):
    prices = np.linspace(price * 0.8, price * 1.2, 50)
    revenues = []

    for p in prices:
        pred = predict_sales(p)
        revenues.append(pred * p)

    prices = np.array(prices)
    revenues = np.array(revenues)

    best_idx = np.argmax(revenues)

    return prices, revenues, prices[best_idx], revenues[best_idx]

# -----------------------------
# RESULTS
# -----------------------------
if run:

    sales = predict_sales(price)
    revenue = sales * price

    prices, revenues, optimal_price, max_revenue = optimize_price(price)

    improvement = ((max_revenue - revenue) / revenue) * 100 if revenue != 0 else 0

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Sales", f"{sales:.2f}")
    col2.metric("Revenue", f"{revenue:.2f}")
    col3.metric("Optimal Price", f"{optimal_price:.2f}")
    col4.metric("Max Revenue", f"{max_revenue:.2f}")

    st.markdown("### Revenue Improvement")
    st.metric("Improvement (%)", f"{improvement:.2f}%")

    # Graph
    st.markdown("### Price Optimization Curve")

    fig, ax = plt.subplots(figsize=(9,4))

    ax.plot(prices, revenues, color="#3B82F6", linewidth=2)
    ax.axvline(optimal_price, linestyle='--', color="#22C55E")

    ax.set_facecolor("#0A0F1C")
    fig.patch.set_facecolor("#0A0F1C")

    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')

    ax.set_xlabel("Price")
    ax.set_ylabel("Revenue")

    st.pyplot(fig)