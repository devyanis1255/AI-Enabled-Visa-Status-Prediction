import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Visa AI", layout="wide")

# ---------------- CACHE LOAD ----------------
@st.cache_data
def load_data():
    df = pd.read_excel("updated_cleaned_H1B_data.xlsx")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

@st.cache_resource
def load_model():
    model = joblib.load("best_model.pkl")
    columns = joblib.load("columns.pkl")
    return model, columns

df = load_data()
model, columns = load_model()

# ---------------- CACHE DROPDOWNS ----------------
@st.cache_data
def get_unique_cached(col):
    if col in df.columns:
        return sorted(df[col].dropna().astype(str).unique())
    return []

# ---------------- CSS ----------------
st.markdown("""
<style>

/* ===== BUTTON ===== */
.stButton>button {
    background: linear-gradient(90deg, #2563eb, #06b6d4);
    color: white !important;
    border-radius: 10px;
    padding: 10px 18px;
    border: none;
    font-weight: 500;
}

.stButton>button:hover {
    transform: scale(1.03);
}

/* ===== TITLE ===== */
h1 {
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧠 Visa AI Dashboard")
page = st.sidebar.radio("Navigate", ["Predict", "Analytics", "History"])

# ---------------- PREPROCESS ----------------
def preprocess(data):
    df_input = pd.DataFrame([data])

    for col in df_input.columns:
        if df_input[col].dtype == "object":
            df_input[col] = df_input[col].astype(str).apply(len)

    for col in columns:
        if col not in df_input:
            df_input[col] = 0

    return df_input[columns]

# ===================== PREDICT =====================
if page == "Predict":

    st.title("🛂 Visa Processing AI")

    colA, colB, colC = st.columns(3)
    colA.metric("Model", "XGBoost")
    colB.metric("Dataset", f"{len(df):,}")
    colC.metric("Status", "Active")

    st.markdown("---")

    REQUIRED_COLS = [
        "case_status",
        "visa_class",
        "pw_wage_level",
        "employer_city",
        "worksite_city"
    ]

    with st.container(border=True):
        col1, col2 = st.columns(2)
        input_data = {}

        # -------- FAST DROPDOWNS --------
        for i, col in enumerate(REQUIRED_COLS):
            values = get_unique_cached(col)

            if values:
                target = col1 if i % 2 == 0 else col2
                with target:
                    input_data[col] = st.selectbox(
                        col.replace("_", " ").title(),
                        values
                    )

        # -------- DATE INPUT --------
        with col1:
            application_date = st.date_input("Application Date")

        with col2:
            decision_date = st.date_input("Decision Date")

        input_data["app_year"] = application_date.year
        input_data["app_month"] = application_date.month
        input_data["dec_year"] = decision_date.year
        input_data["dec_month"] = decision_date.month

        submit = st.button("🚀 Predict Processing Time", use_container_width=True)

    # -------- FAST PREDICTION --------
    if submit:

        with st.spinner("Predicting..."):
            processed = preprocess(input_data)
            pred = model.predict(processed)[0]

        st.success(f"⏱ Estimated: {int(pred)} days")

        st.progress(min(int(pred)/200, 1.0))

        # -------- SAVE HISTORY --------
        hist = pd.DataFrame([input_data])
        hist["prediction"] = pred

        if os.path.exists("history.csv"):
            hist.to_csv("history.csv", mode='a', header=False, index=False)
        else:
            hist.to_csv("history.csv", index=False)

# ===================== ANALYTICS =====================
elif page == "Analytics":

    st.title("📊 Insights")

    if "processing_days" in df.columns:
        st.bar_chart(df["processing_days"])

    if "app_year" in df.columns:
        st.line_chart(df.groupby("app_year")["processing_days"].mean())

# ===================== HISTORY =====================
elif page == "History":

    st.title("📁 History")

    if os.path.exists("history.csv"):
        hist = pd.read_csv("history.csv")
        st.dataframe(hist)
        st.line_chart(hist["prediction"])
    else:
        st.info("No history yet")