#!/usr/bin/env python3
"""
training.py

Usage:
    python training.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# Load dataset
data = pd.read_csv("Data/sample_101_Mental_Health_Lifestyle_Dataset.csv")  # Replace with your file path
print(f"Loaded dataset with {data.shape[0]} rows and {data.shape[1]} columns.")

# Features and target
X = data.drop("Happiness Score", axis=1)
y = data["Happiness Score"]

# Identify numeric columns
numeric_cols = ["Age", "Sleep Hours", "Work Hours per Week", "Screen Time per Day (Hours)", "Social Interaction Score"]

# Convert numeric columns to float to avoid warnings
X[numeric_cols] = X[numeric_cols].astype(float)

# Encode categorical columns
X = pd.get_dummies(X, drop_first=True)

# Scale numeric features
scaler = MinMaxScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Feature selection
selector = SelectKBest(score_func=f_regression, k='all')  # Keep all features
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
X_selected = pd.DataFrame(X_selected, columns=selected_features)

print(f"Selected features: {list(selected_features)}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# Train model
model = Ridge()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"R² score on test set: {r2_score(y_test, y_pred):.3f}")
print(f"Mean Squared Error on test set: {mean_squared_error(y_test, y_pred):.3f}")

# Save model, scaler, and selector
joblib.dump(model, "ridge_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(selector, "feature_selector.pkl")
print("Trained model, scaler, and feature selector saved.")

# Example prediction
sample = X_selected.iloc[0:1]
predicted_happiness = model.predict(sample)[0]
print(f"Sample predicted happiness score: {predicted_happiness:.2f}")
