#!/usr/bin/env python3
"""
testing_accuracy_percent.py

Usage:
    python testing_accuracy_percent.py
"""

import pandas as pd
import joblib
import numpy as np

# -----------------------
# Load trained model, scaler, and feature selector
# -----------------------
model = joblib.load("ridge_model.pkl")
scaler = joblib.load("scaler.pkl")
selector = joblib.load("feature_selector.pkl")
print("Loaded trained model, scaler, and feature selector.")

# -----------------------
# Load testing data
# -----------------------
data = pd.read_csv("Data/testingdata.csv")
print(f"Loaded testing dataset with {data.shape[0]} rows and {data.shape[1]} columns.")

# -----------------------
# Features and target
# -----------------------
X_test = data.drop("Happiness Score", axis=1)
y_test = data["Happiness Score"].values

numeric_cols = ["Age", "Sleep Hours", "Work Hours per Week", "Screen Time per Day (Hours)", "Social Interaction Score"]

# Convert numeric columns to float
X_test[numeric_cols] = X_test[numeric_cols].astype(float)

# Encode categorical columns
X_test = pd.get_dummies(X_test, drop_first=True)

# Align columns with training features
X_test = X_test.reindex(columns=selector.feature_names_in_, fill_value=0)

# Scale numeric features
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# Apply feature selection
X_test_selected = selector.transform(X_test)

# -----------------------
# Predict
# -----------------------
y_pred = model.predict(X_test_selected)

# -----------------------
# Calculate per-sample accuracy
# -----------------------
# Clip negative accuracies to 0%
accuracies = 100 * (1 - np.abs(y_test - y_pred) / y_test)
accuracies = np.clip(accuracies, 0, 100)

# Average accuracy
average_accuracy = np.mean(accuracies)

print("\nPer-sample prediction accuracy (%):")
for i, acc in enumerate(accuracies, 1):
    print(f"Person {i}: {acc:.2f}%")

print(f"\nAverage prediction accuracy: {average_accuracy:.2f}%")
