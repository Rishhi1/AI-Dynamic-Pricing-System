import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Pricing Dashboard", layout="wide")

# -----------------------------
# UI CSS
# -----------------------------
st.markdown("""
<style>
.stApp { background-color: #0A0F1C; color: #E5E7EB; }
[data-testid="stSidebar"] { background-color: #020617; }
.block-container { padding-top: 1.5rem; max-width: 1200px; }
h1 { font-size: 30px !important; font-weight: 600; }

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03);
    padding: 15px;
    border-radius: 12px;
}

div.stButton > button {
    background: #2563EB;
    color: white !important;
    border-radius: 8px;
    height: 42px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD DEFAULT MODEL
# -----------------------------
model = joblib.load("best_model.pkl")
feature_columns = model.feature_names_in_

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("Pricing Controls")

uploaded_file = st.sidebar.file_uploader("Upload CSV")
price = st.sidebar.number_input("Product Price", value=100.0)
run = st.sidebar.button("Run Analysis")

# -----------------------------
# HEADER
# -----------------------------
st.title("Dynamic Pricing Dashboard")
st.caption("Machine learning powered revenue optimization")
st.markdown("---")

# -----------------------------
# DATA HANDLING
# -----------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    target_column = st.selectbox("Select Target Column", df.columns)
else:
    df = None
    target_column = None

# -----------------------------
# PREPROCESSING
# -----------------------------
def preprocess_data(df, target_column):
    df = df.drop_duplicates()
    df = df.fillna(method="ffill")

    y = df[target_column]
    X = df.drop(columns=[target_column])

    X = pd.get_dummies(X, drop_first=True)

    return X, y

# -----------------------------
# AUTO ML TRAINING
# -----------------------------
def train_best_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100),
        "XGBoost": XGBRegressor(n_estimators=200, max_depth=5)
    }

    best_model = None
    best_score = -np.inf
    best_name = ""

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = r2_score(y_test, preds)

        if score > best_score:
            best_score = score
            best_model = model
            best_name = name

    return best_model, X.columns, best_name, best_score

# -----------------------------
# DYNAMIC PREDICTION
# -----------------------------
def predict_sales_dynamic(model, input_df, feature_columns):
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    return max(0, model.predict(input_df)[0])

# -----------------------------
# DEFAULT MODEL FUNCTIONS
# -----------------------------
def predict_sales(price):
    df_input = pd.DataFrame([{"price": price}])
    df_input = df_input.reindex(columns=feature_columns, fill_value=0)
    return max(0, model.predict(df_input)[0])

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

    # -----------------------------
    # DATASET MODE (NEW VERSION 2)
    # -----------------------------
    if df is not None and target_column is not None:

        st.info("Training model on uploaded dataset...")

        X, y = preprocess_data(df, target_column)

        model_dynamic, feature_cols, model_name, model_score = train_best_model(X, y)

        st.success(f"Best Model: {model_name} (R²: {model_score:.3f})")

        sample_row = X.iloc[0:1]

        prices = np.linspace(price * 0.8, price * 1.2, 50)
        revenues = []

        for p in prices:
            temp = sample_row.copy()

            if "price" in temp.columns:
                temp["price"] = p

            pred = predict_sales_dynamic(model_dynamic, temp, feature_cols)
            revenues.append(pred * p)

        prices = np.array(prices)
        revenues = np.array(revenues)

        best_idx = np.argmax(revenues)
        optimal_price = prices[best_idx]
        max_revenue = revenues[best_idx]

        base_sales = predict_sales_dynamic(model_dynamic, sample_row, feature_cols)
        base_revenue = base_sales * price

        improvement = ((max_revenue - base_revenue) / base_revenue) * 100

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Sales", f"{base_sales:.2f}")
        col2.metric("Revenue", f"{base_revenue:.2f}")
        col3.metric("Optimal Price", f"{optimal_price:.2f}")
        col4.metric("Max Revenue", f"{max_revenue:.2f}")

        st.metric("Improvement (%)", f"{improvement:.2f}%")

    # -----------------------------
    # DEFAULT MODE (OLD)
    # -----------------------------
    else:

        sales = predict_sales(price)
        revenue = sales * price

        prices, revenues, optimal_price, max_revenue = optimize_price(price)

        improvement = ((max_revenue - revenue) / revenue) * 100 if revenue != 0 else 0

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Sales", f"{sales:.2f}")
        col2.metric("Revenue", f"{revenue:.2f}")
        col3.metric("Optimal Price", f"{optimal_price:.2f}")
        col4.metric("Max Revenue", f"{max_revenue:.2f}")

        st.metric("Improvement (%)", f"{improvement:.2f}%")

    # -----------------------------
    # GRAPH (COMMON)
    # -----------------------------
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
