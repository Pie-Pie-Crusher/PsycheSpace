import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, RobustScaler

# -----------------------------
# 1️⃣ Load dataset
# -----------------------------
df = pd.read_csv("data/sample_101_Mental_Health_Lifestyle_Dataset copy 2.csv")
print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")

# -----------------------------
# 2️⃣ Define target column
# -----------------------------
target_col = "Happiness Score"

# -----------------------------
# 3️⃣ Handle missing values
# -----------------------------
# Numeric columns (excluding target) → median
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_cols.remove(target_col)  # do not scale target
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Categorical columns → mode
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

# -----------------------------
# 4️⃣ Encode categorical columns (excluding Country and Gender)
# -----------------------------
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
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# -----------------------------
# 7️⃣ Save processed dataset
# -----------------------------
df.to_csv("processed_data.csv", index=False)
print(f"Processed dataset saved to 'processed_data.csv'.")
