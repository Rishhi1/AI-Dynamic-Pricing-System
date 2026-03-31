import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
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
.stApp { background-color: var(--background-color); color: var(--text-color); }
[data-testid="stSidebar"] { background-color: #020617; }
.block-container { padding-top: 1.5rem; max-width: 1200px; }
h1 { font-size: 28px !important; font-weight: 700; }
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
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
# PREPROCESSING
# -----------------------------
def preprocess_data(df, target_column):
    df = df.copy().drop_duplicates().ffill()

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df = df.drop(columns=['date'])

    if target_column not in df.columns:
        return None, None

    y = df[target_column]
    X = pd.get_dummies(df.drop(columns=[target_column]), drop_first=True)

    return X, y

# -----------------------------
# MODEL TRAINING
# -----------------------------
def train_best_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=50, max_depth=6),
    }

    best_model, best_score, best_name = None, -np.inf, ""

    for name, model in models.items():
        model.fit(X_train, y_train)
        score = r2_score(y_test, model.predict(X_test))

        if score > best_score:
            best_score, best_model, best_name = score, model, name

    return best_model, X.columns, best_name, best_score

# -----------------------------
# PREDICTION
# -----------------------------
def predict(model, input_df, feature_columns):
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    return max(0, model.predict(input_df)[0])

# -----------------------------
# CHAT FUNCTIONS
# -----------------------------
def dynamic_prediction_response(query):
    model = st.session_state.get("model")
    sample_row = st.session_state.get("sample_row")
    feature_cols = st.session_state.get("feature_cols")

    if model is None:
        return None

    numbers = re.findall(r"\d+\.?\d*", query)

    if numbers:
        price_val = float(numbers[0])
        temp = sample_row.copy()

        if "price" in temp.columns:
            temp["price"] = price_val

        pred = predict(model, temp, feature_cols)
        revenue = pred * price_val

        return f"For price {price_val}, predicted demand is {pred:.2f} and revenue is {revenue:.2f}"

    return None

def smart_ai_chat(query):
    query = query.lower()

    df = st.session_state.get("df")
    optimal_price = st.session_state.get("optimal_price")
    max_revenue = st.session_state.get("max_revenue")
    base_revenue = st.session_state.get("base_revenue")

    dynamic = dynamic_prediction_response(query)
    if dynamic:
        return dynamic

    if df is not None:
        if "columns" in query:
            return f"Columns: {', '.join(df.columns)}"
        if "rows" in query:
            return f"Dataset has {df.shape[0]} rows"
        if "summary" in query:
            return df.describe().to_string()

    if "optimal price" in query:
        return f"Optimal price is {optimal_price:.2f}"

    if "revenue" in query:
        return f"Base revenue: {base_revenue:.2f}, Max revenue: {max_revenue:.2f}"

    return "Ask about price, revenue, dataset, or predictions."

# -----------------------------
# RESULTS
# -----------------------------
if run and df is not None and target_column is not None:

    st.info("Training model...")

    X, y = preprocess_data(df, target_column)

    if X is None:
        st.error("Invalid target column")
        st.stop()

    model, feature_cols, model_name, model_score = train_best_model(X, y)

    st.success(f"Best Model: {model_name} | R²: {model_score:.3f}")

    sample_row = X.iloc[0:1]

    prices = np.linspace(price * 0.8, price * 1.2, 50)
    revenues = []

    for p in prices:
        temp = sample_row.copy()
        if "price" in temp.columns:
            temp["price"] = p

        pred = predict(model, temp, feature_cols)
        revenues.append(pred * p)

    optimal_price = prices[np.argmax(revenues)]
    max_revenue = max(revenues)

    base_sales = predict(model, sample_row, feature_cols)
    base_revenue = base_sales * price

    # SAVE STATE
    st.session_state.update({
        "model": model,
        "feature_cols": feature_cols,
        "sample_row": sample_row,
        "optimal_price": optimal_price,
        "max_revenue": max_revenue,
        "base_revenue": base_revenue,
        "df": df
    })

    # METRICS
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sales", f"{base_sales:.2f}")
    col2.metric("Revenue", f"{base_revenue:.2f}")
    col3.metric("Optimal Price", f"{optimal_price:.2f}")
    col4.metric("Max Revenue", f"{max_revenue:.2f}")

    # GRAPH
    fig, ax = plt.subplots()
    ax.plot(prices, revenues)
    ax.axvline(optimal_price, linestyle='--')
    st.pyplot(fig)

# -----------------------------
# CHATBOT
# -----------------------------
st.markdown("---")
st.subheader("AI Pricing Assistant")

if "model" not in st.session_state:
    st.info("Run analysis first to enable AI assistant")

else:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    user_input = st.chat_input("Ask something...")

    if user_input:
        reply = smart_ai_chat(user_input)

        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": reply})

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
