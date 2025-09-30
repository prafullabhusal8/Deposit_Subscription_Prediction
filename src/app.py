"""
   pip install -r requirements.txt
   streamlit run src/app.py
"""

import streamlit as st
from load_data import loaded_data
from data_ttt import data_train_test_split
from model_utils import train_models
from ui_utils import apply_custom_css
from evaluate_model import evaluate_models
from predict_utils import preprocess_user_input

# -------------------------
# Page Config & CSS
# -------------------------
st.set_page_config(page_title="Model", page_icon="⚡", layout="wide")
apply_custom_css()

# -------------------------
# Title
# -------------------------
st.markdown('<h1 class="main-header">⚡ Model</h1>', unsafe_allow_html=True)
st.markdown("Train models and make predictions")

# -------------------------
# Data
# -------------------------
df_n = loaded_data(r"C:\Users\bhusa\Downloads\Deposit_Subscription_Prediction\data\Deposit_Subscription_Prediction.csv")

(x_train_pca, y_train), (x_test_pca, y_test), pca = data_train_test_split(df_n)

# -------------------------
# Sidebar: Model selection
# -------------------------
st.sidebar.header("⚙️ Model Settings")
model_choice = st.sidebar.selectbox(
    "Choose a model type:",
    ["Decision Tree", "Random Forest", "XGBoost"]
)

model_type_map = {
    "Decision Tree": "decision_tree",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost"
}

# Train models based on selection
models = train_models((x_train_pca, y_train), model_type=model_type_map[model_choice])

performance = evaluate_models((x_test_pca, y_test), models)

# -------------------------
# Sidebar: Performance
# -------------------------
st.sidebar.subheader("📊 Model Performance")
for task, metrics in performance.items():
    st.sidebar.markdown(f"**{task}**")
    st.sidebar.write(f"Accuracy: {metrics['Accuracy']:.3f}")
    st.sidebar.write(f"Precision: {metrics['Precision']:.3f}")
    st.sidebar.write(f"Recall: {metrics['Recall']:.3f}")
    st.sidebar.write(f"F1 Score: {metrics['F1 Score']:.3f}")
    st.sidebar.subheader("Confusion Matrix")
    st.sidebar.table(metrics["Confusion Matrix"])
    st.sidebar.markdown("---")

# -------------------------
# User Input Prediction
# -------------------------
st.markdown("## 🔮 Make a Prediction")

with st.form("user_input_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    balance = st.number_input("Balance", value=1000)
    duration = st.number_input("Duration", value=100)
    campaign = st.number_input("Campaign", value=1)
    pdays = st.number_input("Pdays", value=-1)
    previous = st.number_input("Previous", value=0)

    job = st.selectbox("Job", df_n.filter(like="job_").columns.str.replace("job_", ""))
    marital = st.selectbox("Marital", df_n.filter(like="marital_").columns.str.replace("marital_", ""))
    education = st.selectbox("Education", df_n.filter(like="education_").columns.str.replace("education_", ""))
    default = st.selectbox("Default", df_n.filter(like="default_").columns.str.replace("default_", ""))
    housing = st.selectbox("Housing Loan", df_n.filter(like="housing_").columns.str.replace("housing_", ""))
    loan = st.selectbox("Personal Loan", df_n.filter(like="loan_").columns.str.replace("loan_", ""))
    contact = st.selectbox("Contact", df_n.filter(like="contact_").columns.str.replace("contact_", ""))
    month = st.selectbox("Month", df_n.filter(like="month_").columns.str.replace("month_", ""))
    poutcome = st.selectbox("Previous Outcome", df_n.filter(like="poutcome_").columns.str.replace("poutcome_", ""))

    submit_btn = st.form_submit_button("Predict")

if submit_btn:
    user_data = {
        "age": [age],
        "balance": [balance],
        "duration": [duration],
        "campaign": [campaign],
        "pdays": [pdays],
        "previous": [previous],
        "job": [job],
        "marital": [marital],
        "education": [education],
        "default": [default],
        "housing": [housing],
        "loan": [loan],
        "contact": [contact],
        "month": [month],
        "poutcome": [poutcome],
    }

    # Preprocess input (encoding + PCA using trained pca)
    user_pca = preprocess_user_input(user_data, df_n, pca)

    # Predict
    prediction = models["classification"].predict(user_pca)[0]
    label = "YES" if prediction == 1 else "NO"
    st.success(f"✅ Prediction: The customer is **{label}** to subscription.")
