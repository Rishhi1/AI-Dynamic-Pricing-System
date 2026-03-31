import streamlit as st
import pandas as pd
import numpy as np
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
/* App background */
.stApp {
    background-color: var(--background-color);
    color: var(--text-color);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #020617;
}

/* Layout */
.block-container {
    padding-top: 1.5rem;
    max-width: 1200px;
}

/* Title */
h1 {
    font-size: 28px !important;
    font-weight: 700;
}

/* Metrics */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    padding: 15px;
    border-radius: 12px;
}

/* Button */
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
# SIDEBAR
# -----------------------------
st.sidebar.title("Pricing Controls")

uploaded_file = st.sidebar.file_uploader("Upload CSV")
price = st.sidebar.number_input("Base Price", value=100.0)
run = st.sidebar.button("Run Analysis")

# -----------------------------
# HEADER
# -----------------------------
st.title("AI Dynamic Pricing System")
st.caption("AI-powered revenue optimization engine")
st.markdown("---")

# -----------------------------
# DATA HANDLING
# -----------------------------
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8')
    except:
        try:
            df = pd.read_csv(uploaded_file, encoding='latin1')
        except:
            df = pd.read_excel(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    target_column = st.selectbox("Select Target Column", df.columns)
else:
    df = None
    target_column = None

# -----------------------------
# PREPROCESSING (CACHED)
# -----------------------------
@st.cache_data(show_spinner=False)
def preprocess_data(df, target_column):

    df = df.copy()
    df = df.drop_duplicates()
    df = df.ffill()

    # Handle date column
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df = df.drop(columns=['date'])

    if target_column not in df.columns:
        return None, None

    y = df[target_column]
    X = df.drop(columns=[target_column])

    # Convert categorical
    X = pd.get_dummies(X, drop_first=True)

    return X, y

# -----------------------------
# MODEL TRAINING (CACHED)
# -----------------------------
@st.cache_resource
def train_best_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=50, max_depth=6),
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
# PREDICTION
# -----------------------------
def predict(model, input_df, feature_columns):
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    return max(0, model.predict(input_df)[0])

# -----------------------------
# RESULTS
# -----------------------------
if run:

    if df is None:
        st.error("Please upload a dataset first")
    
    else:
        st.info("Training model...")

        X, y = preprocess_data(df, target_column)

        if X is None:
            st.error("Invalid target column")
            st.stop()

        model, feature_cols, model_name, model_score = train_best_model(X, y)

        st.success(f"Best Model: {model_name} | R²: {model_score:.3f}")

        # Use first row as baseline
        sample_row = X.iloc[0:1]

        prices = np.linspace(price * 0.8, price * 1.2, 50)
        revenues = []

        for p in prices:
            temp = sample_row.copy()

            if "price" in temp.columns:
                temp["price"] = p

            pred = predict(model, temp, feature_cols)
            revenues.append(pred * p)

        prices = np.array(prices)
        revenues = np.array(revenues)

        best_idx = np.argmax(revenues)
        optimal_price = prices[best_idx]
        max_revenue = revenues[best_idx]

        base_sales = predict(model, sample_row, feature_cols)
        base_revenue = base_sales * price

        improvement = ((max_revenue - base_revenue) / base_revenue) * 100 if base_revenue != 0 else 0

        # -----------------------------
        # KPI METRICS
        # -----------------------------
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Sales", f"{base_sales:.2f}")
        col2.metric("Revenue", f"{base_revenue:.2f}")
        col3.metric("Optimal Price", f"{optimal_price:.2f}")
        col4.metric("Max Revenue", f"{max_revenue:.2f}")

        st.metric("Improvement (%)", f"{improvement:.2f}%")

        # -----------------------------
        # GRAPH
        # -----------------------------
        st.markdown("### Price Optimization Curve")

        fig, ax = plt.subplots(figsize=(9,4))

        ax.plot(prices, revenues)
        ax.axvline(optimal_price, linestyle='--')

        ax.set_facecolor("#0A0F1C")
        fig.patch.set_facecolor("#0A0F1C")

        ax.tick_params(colors='white')
        ax.set_xlabel("Price", color='white')
        ax.set_ylabel("Revenue", color='white')

        st.pyplot(fig)
