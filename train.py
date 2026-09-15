import joblib, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

SHOW_PLOTS = False

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Basic information
print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumns")
print(df.columns.tolist())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

print("\nTarget Distribution")
print(df["Churn"].value_counts())

# Inspecting why the TotalCharges column is str datatype
print("\nTotal Charges unique problematic values:")
print(df["TotalCharges"].value_counts().tail(10))

print("\nBlank TotalCharges:")
print((df["TotalCharges"].str.strip() == "").sum())

# Converting the column to Numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce") # errors = "coerce" -> if you find something isn't a valid number , convert it to NaN

# verifying the above change
print(df["TotalCharges"].dtype)
print("Missing TotalCharges:", df["TotalCharges"].isnull().sum())

print(
    df[df["TotalCharges"].isnull()]                           # why are we doing this -> we dont want to blindly do
    [["tenure", "MonthlyCharges", "TotalCharges", "Churn"]]   # df["TotalCharges"].fillna(...)
)


# we will be treating TotalCharges as 0 , as the tenure is 0
# TotalCharges = MonthlyCharges * tenure

df["TotalCharges"] = df["TotalCharges"].fillna(0)

# reverifying missing values
print("Missing values after cleaning:")
print(df.isnull().sum().sum())

print("\nTotalCharges dtype:")
print(df["TotalCharges"].dtype)


# ======================
# 3. Target Distribution
# ======================

churn_counts = df["Churn"].value_counts()

print("\nChurn Distribution:")
print(churn_counts)

if SHOW_PLOTS:
    plt.figure(figsize=(6,4))
    plt.bar(
        churn_counts.index,
        churn_counts.values
    )

    plt.xlabel("Churn")
    plt.ylabel("Number of Custoemrs")
    plt.title("Customer Churn Distribution")
    plt.show()


# =========================
# 4. Churn Rate by Contract
# =========================

contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Rate by Contract:")
print(contract_churn)

if SHOW_PLOTS:
    plt.figure(figsize = (8,5))
    contract_churn["Yes"].plot(
        kind="bar"
    )

    plt.xlabel("Contract type")
    plt.ylabel("Churn Rate (%)")
    plt.title("Churn Rate by Contract type")
    plt.xticks(rotation=0)
    plt.show()


# ==================
# 5. Tenure vs Churn
# ==================

tenure_churn = df.groupby("Churn")["tenure"].mean()

print("\nAverage Tenure by Churn:")
print(tenure_churn)
if SHOW_PLOTS:
    df.boxplot(column="tenure", by="Churn")
    plt.title("Tenure Distribution by Churn")
    plt.suptitle("")
    plt.xlabel("Churn")
    plt.ylabel("Tenure (Months)")
    plt.show()


# ===========================
# 6. Monthly Charges by Churn
# ===========================

monthly_charges_churn = df.groupby("Churn")["MonthlyCharges"].mean()

print("\nAverage Monthly Charges by Churn:")
print(monthly_charges_churn)

if SHOW_PLOTS:
    df.boxplot(column = "MonthlyCharges", by= "Churn", figsize=(7,5))

    plt.title("Monthly Charges Distribution by Churn")
    plt.suptitle("")
    plt.xlabel("Churn")
    plt.ylabel("Monthly Charges")
    plt.show()

# =================================
# 7. Churn Rate by Internet Service
# =================================

internet_churn = pd.crosstab(
    df["InternetService"],
    df["Churn"],
    normalize = "index"
) * 100

print("\n Churn Rate by Internet Service:")
print(internet_churn)

plt.figure(figsize=(7,5))

internet_churn["Yes"].plot(
    kind = "bar"
)
if SHOW_PLOTS:
    plt.title("Churn Rate by Internet Service")
    plt.xlabel("Internet Service")
    plt.ylabel("Churn Rate (%)")
    plt.xticks(rotation=0)
    plt.show()


# =========================
# 8. Churn Rate by Tech Support
# =========================

tech_support_churn = pd.crosstab(
    df["TechSupport"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Rate by Tech Support:")
print(tech_support_churn)

if SHOW_PLOTS:
    plt.figure(figsize=(7, 5))

    tech_support_churn["Yes"].plot(
        kind="bar"
    )

    plt.title("Churn Rate by Tech Support")
    plt.xlabel("Tech Support")
    plt.ylabel("Churn Rate (%)")
    plt.xticks(rotation=0)
    plt.show()


# ======================
# 9. Prepare Data for ML
# ======================

# Remove customer ID
df = df.drop("customerID", axis=1)

# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

print("\nFeature columns:")
print(X.columns.tolist())

print("\nUnique target values BEFORE encoding:")
print(y.unique())

print("\nTarget value counts BEFORE encoding:")
print(y.value_counts())

y = (y == "Yes").astype(int)

print("\nUnique target values AFTER encoding:")
print(y.unique())

print("\nTarget value counts AFTER encoding:")
print(y.value_counts())


# ====================
# 10. Train-Test Split
# ====================

X_train, X_test , y_train , y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTest target distribution:")
print(y_test.value_counts())


# =========================
# 11. Feature Preprocessing
# =========================
numerical_featuring = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

categorical_features = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]

preprocessor = ColumnTransformer(
    transformers = [
        (
            "num",
            StandardScaler(),
            numerical_featuring
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown = "ignore"),
            categorical_features
        )
    ]
)


# =======================
# 12. Logistic Regression
# =======================

logistic_model = Pipeline(
    steps=[
        ("preproceessor", preprocessor),
        ("classsifer", LogisticRegression(max_iter=1000))
    ]
)

# Train the model
logistic_model.fit(X_train, y_train)

# Predictions
y_pred = logistic_model.predict(X_test)

#Probability of churn = Yes
y_prob = logistic_model.predict_proba(X_test)[:,1]

# ====================
# 13. Model Evaluation
# ====================

print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =================
# 14. Random Forest
# =================

random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)

# Train the model
random_forest_model.fit(X_train , y_train)

# Predictions
rf_pred = random_forest_model.predict(X_test)

# Churn Probabilities
rf_prob = random_forest_model.predict_proba(X_test)[:, 1]

# ============================
# 15. Random Forest Evaluation
# ============================

print("\nRandom Forest Evaluation:")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred))
print("Recall:", recall_score(y_test, rf_pred))
print("F1 Score:", f1_score(y_test, rf_pred))
print("ROC-AUC:", roc_auc_score(y_test, rf_prob))

print("\nClassification Report:")
print(classification_report(y_test, rf_pred))


# =====================
# 16. Gradient Boosting
# =====================

gradient_boosting_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            )
        )
    ]
)

# Train the model
gradient_boosting_model.fit(X_train, y_train)

# Predictions
gb_pred = gradient_boosting_model.predict(X_test)

# Churn probabilities
gb_prob = gradient_boosting_model.predict_proba(X_test)[:,1]

# ================================
# 17. Gradient Boosting Evaluation
# ================================

print("\nGradient Boosting Evaluation:")
print("Accuracy:", accuracy_score(y_test, gb_pred))
print("Precision:", precision_score(y_test, gb_pred))
print("Recall:", recall_score(y_test, gb_pred))
print("F1 Score:", f1_score(y_test, gb_pred))
print("ROC-AUC:", roc_auc_score(y_test, gb_prob))

print("\nClassification Report:")
print(classification_report(y_test, gb_pred))


# =========================
# 18. Logistic Regression
#     Hyperparameter Tuning
# =========================

logistic_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

# Hyperparameters to test
param_grid = {
    "classifier__C": [0.01, 0.1, 1, 10, 100]
}

grid_search = GridSearchCV(
    estimator = logistic_pipeline,
    param_grid=param_grid,
    cv = 5,
    scoring = "roc_auc",
    n_jobs = -1
)

# Train using cross-validation
grid_search.fit(X_train , y_train)

print("\n Best Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation ROC_AUC:")
print(grid_search.best_score_)


# ========================
# 19. Evaluate Tuned Model
# ========================

best_logistic_model = grid_search.best_estimator_

tuned_pred = best_logistic_model.predict(X_test)
tuned_prob = best_logistic_model.predict_proba(X_test)[:, 1]

print("\nTuned Logistic Regression Evaluation:")
print("Accuracy:", accuracy_score(y_test, tuned_pred))
print("Precision:", precision_score(y_test, tuned_pred))
print("Recall:", recall_score(y_test, tuned_pred))
print("F1 Score:", f1_score(y_test, tuned_pred))
print("ROC-AUC:", roc_auc_score(y_test, tuned_prob))

print("\nClassification Report:")
print(classification_report(y_test, tuned_pred))


# ==========================
# 20. Threshold Optimization
# ==========================


thresholds = np.arange(0.2, 0.61, 0.05)

print("\nThreshold Analysis:")
print("-" * 65)
print(f"{"Threshold":<12}{"precision":<15}{"Recall":<15}{"F1 Score":<15}")
print("-" * 65)

for threshold in thresholds :
    threshold_pred = (tuned_prob >= threshold).astype(int)

    precision = precision_score(y_test, threshold_pred)
    recall = recall_score(y_test , threshold_pred)
    f1 = f1_score(y_test, threshold_pred)

    print(
        f"{threshold:<12.2f}"
        f"{precision:<15.3f}"
        f"{recall:<15.3f}"
        f"{f1:<15.3f}"
    )

# =========================
# 21. Final Churn Threshold
# =========================

FINAL_THRESHOLD = 0.25

final_pred = (tuned_prob >= FINAL_THRESHOLD).astype(int)

print("\nFinal Model Evaluation:")
print("Threshold:", FINAL_THRESHOLD)
print("Accuracy:", accuracy_score(y_test, final_pred))
print("Precision:", precision_score(y_test, final_pred))
print("Recall:", recall_score(y_test, final_pred))
print("F1 Score:", f1_score(y_test, final_pred))
print("ROC-AUC:", roc_auc_score(y_test, tuned_prob))

print("\nClassification Report:")
print(classification_report(y_test, final_pred))

cm = confusion_matrix(y_test, final_pred)

print("\nConfusion Matrix:")
print(cm)


# ==========================
# 22. Threshold Selection
#     Using Cross-Validation
# ==========================


cv_prob = cross_val_predict(
    best_logistic_model,
    X_train,
    y_train,
    cv = 5,
    method="predict_proba",
    n_jobs=1
)[:,1]

thresholds = np.arange(0.20, 0.61, 0.05)

best_threshold = None
best_f1 = 0

print("\nCross-Validation Threshold Analysis:")
print("-" * 65)
print(f"{'Threshold':<12}{'Precision':<15}{'Recall':<15}{'F1 Score':<15}")
print("-" * 65)


for threshold in thresholds:
    cv_pred = (cv_prob >= threshold).astype(int)

    precision = precision_score(y_train, cv_pred)
    recall = recall_score(y_train, cv_pred)
    f1 = f1_score(y_train , cv_pred)

    print(
        f"{threshold:<12.2f}"
        f"{precision:<15.3f}"
        f"{recall:<15.3f}"
        f"{f1:<15.3f}"
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print("\nBest Threshold:")
print(best_threshold)

print("\nBest Cross-Validation F1 Score:")
print(best_f1)


# =========================
# 23. Final Test Evaluation
# =========================

final_threshold = best_threshold

final_pred = (tuned_prob >= final_threshold).astype(int)

print("\nFinal Test Evaluation:")
print("Threshold:", final_threshold)
print("Accuracy:", accuracy_score(y_test, final_pred))
print("Precision:", precision_score(y_test, final_pred))
print("Recall:", recall_score(y_test, final_pred))
print("F1 Score:", f1_score(y_test, final_pred))
print("ROC-AUC:", roc_auc_score(y_test, tuned_prob))

print("\nClassification Report:")
print(classification_report(y_test, final_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, final_pred))


# ====================
# 24. Save Final Model
# ====================

os.makedirs("models", exist_ok = True)

joblib.dump(
    best_logistic_model,
    "models/churn_model.joblib"
)

print("\nModel saved successfully!")
print("Path: models/churn_model.joblib")

# Save the classification threshold
joblib.dump(
    final_threshold,
    "models/churn_threshold.joblib"
)
print("Threshold saved successfully!")
print("THreshold:", final_threshold)


