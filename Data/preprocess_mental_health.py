#!/usr/bin/env python3
"""
preprocess_mental_health.py

Description:
This script preprocesses a dataset for predicting mental health conditions.
It performs the following steps:
- Handles missing values
- Encodes categorical (non-numerical) columns
- Flags obvious outliers
- Normalizes numeric values

Usage:
    python preprocess_mental_health.py --input raw_data.csv --output processed_data.csv --outliers outliers_report.csv
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import argparse

# -----------------------
# Parse command-line arguments
# -----------------------
parser = argparse.ArgumentParser(description="Preprocess mental health dataset")
parser.add_argument('--input', required=True, help='Path to input CSV file')
parser.add_argument('--output', required=True, help='Path to save processed CSV file')
parser.add_argument('--outliers', required=True, help='Path to save flagged outliers CSV file')
args = parser.parse_args()

# -----------------------
# Load the dataset
# -----------------------
df = pd.read_csv(args.input)
print(f"Initial dataset shape: {df.shape}")

# -----------------------
# 1. Handling Missing Values
# -----------------------
# Numeric columns: fill missing with median
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Categorical columns: fill missing with mode
categorical_cols = df.select_dtypes(include='object').columns
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

print("Missing values handled.")

# -----------------------
# 2. Encoding Categorical Columns
# -----------------------
# Using Label Encoding for simplicity
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

print("Categorical columns encoded.")

# -----------------------
# 3. Flag Obvious Outliers
# -----------------------
# Using IQR method: values outside 1.5 * IQR are flagged
outliers_dict = {}
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outlier_rows = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    if not outlier_rows.empty:
        outliers_dict[col] = outlier_rows.index.tolist()

# Save flagged outliers
outlier_records = []
for col, rows in outliers_dict.items():
    for row in rows:
        outlier_records.append({'row_index': row, 'column': col, 'value': df.at[row, col]})

outliers_df = pd.DataFrame(outlier_records)
outliers_df.to_csv(args.outliers, index=False)
print(f"Outliers flagged and saved to {args.outliers}")

# -----------------------
# 4. Normalize Numeric Columns
# -----------------------
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
print("Numeric columns normalized.")

# -----------------------
# Save Processed Data
# -----------------------
df.to_csv(args.output, index=False)
print(f"Processed dataset saved to {args.output}")
