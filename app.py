import streamlit as st
import pandas as pd
import joblib


# =====================
# 1. Page Configuration
# =====================

st.set_page_config(
    page_title = "Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# =============
# 2. Load Model
# =============

@st.cache_resource
def load_model():
    model = joblib.load("models/churn_model.joblib")
    threshold = joblib.load("models/churn_threshold.joblib")

    return model, threshold

model, threshold = load_model()


# =============
# 3. App Header
# =============

st.title("📊 Customer Churn Predictor")
st.write(
    "Predict whether a customer is likely to churn based on"
    "their account and service information"
)

st.divider()


# =======================
# 4. Customer Information
# =======================

st.subheader("Customer Information")

col1 , col2, col3 = st.columns(3)

with col1 :
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func = lambda x: "Yes" if x == 1 else "No"
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

with col2 :
    tenure = st.number_input(
        "Tenure (Months)",
        min_value = 0,
        max_value = 100,
        value = 12
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fibre Optic", "No"]
    )

with col3 :
    contract = st.selectbox(
        "Contract",
        ["Month-to-Month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed Check",
            "Bank transfer (automatic)",
            "Credit Card (automatic)"
        ]
    )


# ===========
# 5. Services
# ===========

st.subheader("Service")

col1, col2, col3 = st.columns(3)

with col1:
    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

with col2:
    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

with col3:
    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )


# ======================
# 6. Billing Information
# ======================

st.subheader("Billing Information")

col1, col2 = st.columns(2)

with col1 :
    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value = 0.0,
        max_value = 200.0,
        value= 70.0,
        step = 0.5
    )

with col2 :
    total_charges = st.number_input(
        "Monthly Charges",
        min_value = 0.0,
        max_value = 10000.0,
        value= 840.0,
        step = 10.0
    )


# =========================
# 7. Create Input DataFrame
# =========================

customer_data = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [senior_citizen],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "PhoneService": [phone_service],
    "MultipleLines": [multiple_lines],
    "InternetService": [internet_service],
    "OnlineSecurity": [online_security],
    "OnlineBackup": [online_backup],
    "DeviceProtection": [device_protection],
    "TechSupport": [tech_support],
    "StreamingTV": [streaming_tv],
    "StreamingMovies": [streaming_movies],
    "Contract": [contract],
    "PaperlessBilling": [paperless_billing],
    "PaymentMethod": [payment_method],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges],
})


# =============
# 8. Prediction
# =============

st.divider()

if st.button("🔮 Predict Churn", use_container_width=True):
    churn_probability = model.predict_proba(customer_data)[0,1]

    prediction = int(churn_probability >= threshold)

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Churn Probability",
            f"{churn_probability:.1%}"
        )

    with col2:
        st.metric(
            "Decision Threshold",
            f"{threshold:.0%}"
        )

    if prediction == 1 :
        st.error(
            "⚠️ High Churn Risk — Customer is likely to churn."
        )
        st.write(
            "Consider proactive customer retention measures "
            "such as personalized offers or support."
        )
    else:
        st.success(
            "✅ Low Churn Risk — Customer is unlikely to churn."
        )
        st.write(
            "The customer currently shows a relatively low "
            "likelihood of churn."
        )
