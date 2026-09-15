import streamlit as st
import pandas as pd
import joblib

# =========================
# Custom Styling
# =========================

st.markdown("""
<style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #9ca3af;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    .risk-high {
        padding: 18px;
        border-radius: 10px;
        background-color: #3f2020;
        border: 1px solid #7f1d1d;
        margin-top: 20px;
    }

    .risk-low {
        padding: 18px;
        border-radius: 10px;
        background-color: #17351f;
        border: 1px solid #166534;
        margin-top: 20px;
    }

</style>
""", unsafe_allow_html=True)

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

st.markdown(
    '<div class="main-title">📊 Customer Churn Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered customer churn risk prediction using machine learning.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =======================
# 4. Customer Information
# =======================

st.markdown(
    '<div class="section-title">👤 Customer Information</div>',
    unsafe_allow_html=True
)

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

st.markdown(
    '<div class="section-title">🛠️ Services</div>',
    unsafe_allow_html=True
)

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

st.markdown(
    '<div class="section-title">💳 Billing Information</div>',
    unsafe_allow_html=True
)

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
        "Total Charges",
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

    st.markdown(
        '<div class="section-title">🔮 Prediction Result</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Churn Probability",
            f"{churn_probability:.1%}"
        )
        st.progress(
            float(churn_probability),
            text=f"Churn probability: {churn_probability:.1%}"
        )

    with col2:
        st.metric(
            "Decision Threshold",
            f"{threshold:.0%}"
        )

    if prediction == 1:

        st.markdown(
            '<div class="risk-high">'
            '<h3>⚠️ High Churn Risk</h3>'
            '<p>This customer is likely to churn based on the model prediction.</p>'
            '</div>',
            unsafe_allow_html=True
        )

        st.write(
            "Consider proactive retention measures such as "
            "personalized offers, customer support, or contract incentives."
        )

    else:

        st.markdown(
            '<div class="risk-low">'
            '<h3>✅ Low Churn Risk</h3>'
            '<p>This customer is unlikely to churn based on the model prediction.</p>'
            '</div>',
            unsafe_allow_html=True
        )

        st.write(
            "The customer currently shows a relatively low "
            "likelihood of churn."
        )
