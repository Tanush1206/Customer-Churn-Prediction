# 📊 Customer Churn Prediction

An end-to-end machine learning project that predicts whether a telecom customer is likely to churn based on their demographic information, services, contract details, and billing information.

The project includes data preprocessing, exploratory data analysis, model comparison, hyperparameter tuning, cross-validation based threshold selection, model explainability, and a deployed Streamlit web application.

## 🚀 Live Demo

👉 https://customer-churn-prediction-q8sfu8ah5e2ylzz6smxjzh.streamlit.app/

## 🐙 GitHub Repository

👉 https://github.com/Tanush1206/Customer-Churn-Prediction

---

## 🎯 Project Objective

Customer churn is an important business problem for subscription-based companies.

The goal of this project is to:

- Predict whether a customer is likely to churn.
- Identify factors associated with higher churn risk.
- Prioritize recall so that potentially churning customers are less likely to be missed.
- Provide an easy-to-use interface for making predictions on individual customers.

---

## 📁 Dataset

The project uses the **IBM Telco Customer Churn dataset** from Kaggle.

The dataset contains information about:

- Customer demographics
- Account information
- Services subscribed
- Contract type
- Payment method
- Monthly charges
- Total charges
- Customer churn status

### Dataset

**Rows:** 7,043
**Features:** 19 after preprocessing/removing customer ID
**Target:** `Churn`

Target distribution:

- No Churn: 5,174
- Churn: 1,869

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Matplotlib
- Git & GitHub

---

## 🔄 Machine Learning Pipeline

The project follows an end-to-end machine learning workflow:

### 1. Data Cleaning

- Converted `TotalCharges` from string to numeric.
- Handled the 11 blank `TotalCharges` values.
- Removed the `customerID` column.
- Converted the target variable into binary format.

### 2. Train-Test Split

The dataset was split into:

- 80% training data
- 20% test data

Stratified splitting was used to preserve the churn distribution.

### 3. Feature Preprocessing

Numerical features were standardized using `StandardScaler`.

Categorical features were encoded using:

```python
OneHotEncoder(handle_unknown="ignore")
