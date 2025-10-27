import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import OneHotEncoder, RobustScaler

# -----------------------------
# 1️⃣ Load dataset
# -----------------------------
df = pd.read_csv("Data/sample_101_Mental_Health_Lifestyle_Dataset.csv")  # Replace with your CSV filename
print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")

# -----------------------------
# 2️⃣ Handle missing values
# -----------------------------
# Fill numeric columns with median
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Fill categorical columns with mode (most frequent)
categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

# -----------------------------
# 3️⃣ Encode categorical columns
# -----------------------------
# OneHotEncode all categorical columns except the target ('Mental Health Condition')
target_col = "Mental Health Condition"
cat_cols_to_encode = [col for col in categorical_cols if col != target_col]

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded = pd.DataFrame(
    encoder.fit_transform(df[cat_cols_to_encode]),
    columns=encoder.get_feature_names_out(cat_cols_to_encode),
    index=df.index
)

# Drop original categorical columns that were encoded
df = df.drop(columns=cat_cols_to_encode)
# Concatenate encoded columns
df = pd.concat([df, encoded], axis=1)

# -----------------------------
# 4️⃣ Flag obvious outliers
# -----------------------------
outlier_flags = pd.DataFrame(index=df.index)
for col in numeric_cols:
    # IQR method
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outlier_flags[f"{col}_outlier"] = (df[col] < lower) | (df[col] > upper)
# Add combined row-level outlier flag
outlier_flags["any_outlier"] = outlier_flags.any(axis=1)

# Save outlier report
outlier_flags.to_csv("outliers_report.csv", index=False)
print(f"Outlier flags saved to 'outliers_report.csv'.")

# -----------------------------
# 5️⃣ Normalize numeric columns
# -----------------------------
scaler = RobustScaler()  # RobustScaler is resistant to outliers
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# -----------------------------
# 6️⃣ Save processed dataset
# -----------------------------
df.to_csv("processed_data.csv", index=False)
print(f"Processed dataset saved to 'processed_data.csv'.")

