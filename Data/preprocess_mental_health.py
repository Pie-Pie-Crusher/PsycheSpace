import pandas as pd
import numpy as np
<<<<<<< HEAD
from sklearn.preprocessing import OneHotEncoder, RobustScaler
=======
from scipy import stats
from sklearn.preprocessing import LabelEncoder, RobustScaler
>>>>>>> 3acf77fede7ef015b557de87ba3649f8136bde80

# -----------------------------
# 1️⃣ Load dataset
# -----------------------------
<<<<<<< HEAD
df = pd.read_csv("data/sample_101_Mental_Health_Lifestyle_Dataset copy 2.csv")
=======
df = pd.read_csv("Data/sample_101_Mental_Health_Lifestyle_Dataset copy 2.csv")
>>>>>>> 3acf77fede7ef015b557de87ba3649f8136bde80
print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
print(f"Columns: {df.columns.tolist()}")

# -----------------------------
# 2️⃣ Define target column
# -----------------------------
target_col = "Happiness Score"

# -----------------------------
# 3️⃣ Handle missing values
# -----------------------------
# Numeric columns (excluding target) → median
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if target_col in numeric_cols:
    numeric_cols.remove(target_col)  # do not process target

# Fill numeric columns with median
for col in numeric_cols:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"Filled {col} missing values with median: {median_val}")

# Categorical columns → mode
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    if df[col].isnull().any():
        mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
        df[col] = df[col].fillna(mode_val)
        print(f"Filled {col} missing values with mode: {mode_val}")

# -----------------------------
# 4️⃣ Encode categorical columns (excluding Country and Gender)
# -----------------------------
<<<<<<< HEAD
exclude_cols = ["Country", "Gender"]  # columns to exclude from encoding
categorical_to_encode = [col for col in categorical_cols if col not in exclude_cols]

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded = pd.DataFrame(
    encoder.fit_transform(df[categorical_to_encode]),
    columns=encoder.get_feature_names_out(categorical_to_encode),
    index=df.index
)

# Drop original encoded columns and concatenate encoded
df = df.drop(columns=categorical_to_encode)
df = pd.concat([df, encoded], axis=1)
=======
# Use Label Encoding for categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Drop original categorical columns after encoding
df = df.drop(columns=categorical_cols)

# Get updated numeric columns list after encoding
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if target_col in numeric_cols:
    numeric_cols.remove(target_col)  # do not process target
>>>>>>> 3acf77fede7ef015b557de87ba3649f8136bde80

# -----------------------------
# 5️⃣ Flag obvious outliers for numeric features (excluding target)
# -----------------------------
outlier_flags = pd.DataFrame(index=df.index)
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outlier_flags[f"{col}_outlier"] = (df[col] < lower) | (df[col] > upper)

outlier_flags["any_outlier"] = outlier_flags.any(axis=1)
outlier_flags.to_csv("outliers_report.csv", index=False)
print(f"Outlier flags saved to 'outliers_report.csv'.")

# -----------------------------
# 6️⃣ Normalize numeric features (excluding target)
# -----------------------------
scaler = RobustScaler()
# Create a copy of the dataframe to preserve original values
df_normalized = df.copy()
df_normalized[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# -----------------------------
# 7️⃣ Save processed dataset
# -----------------------------
df_normalized.to_csv("processed_data.csv", index=False)
print(f"Processed dataset saved to 'processed_data.csv' with shape: {df_normalized.shape}")
print(f"Features for model (excluding Country and Gender): {[col for col in df_normalized.columns if col not in ['Country_encoded', 'Gender_encoded', target_col]]}")
print(f"Target variable: {target_col}")
