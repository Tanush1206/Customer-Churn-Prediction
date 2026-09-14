import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

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
