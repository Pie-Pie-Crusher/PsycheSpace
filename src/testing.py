#!/usr/bin/env python3
"""
test_model.py

Usage:
    python test_model.py
"""

import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_squared_error

# -----------------------
# Load saved model, scaler, and selector
# -----------------------
model = joblib.load("ridge_model.pkl")
scaler = joblib.load("scaler.pkl")
selector = joblib.load("feature_selector.pkl")
print("Loaded trained model, scaler, and feature selector.")

# -----------------------
# Load testing dataset
# -----------------------
test_data = pd.read_csv("Data/testingdata.csv")  # Replace with your testing CSV path
print(f"Loaded testing dataset with {test_data.shape[0]} rows and {test_data.shape[1]} columns.")

# -----------------------
# Features and target
# -----------------------
X_test = test_data.drop("Happiness Score", axis=1)
y_test = test_data["Happiness Score"]

numeric_cols = ["Age", "Sleep Hours", "Work Hours per Week", "Screen Time per Day (Hours)", "Social Interaction Score"]

# Convert numeric columns to float
X_test[numeric_cols] = X_test[numeric_cols].astype(float)

# Encode categorical columns (same as training)
X_test = pd.get_dummies(X_test, drop_first=True)

# Scale numeric features
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# Keep only the features selected by the model
X_test_selected = X_test[selector.get_feature_names_out()]

# -----------------------
# Predict
# -----------------------
y_pred = model.predict(X_test_selected)

# -----------------------
# Evaluate
# -----------------------
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"R² score on test set: {r2:.3f}")
print(f"Mean Squared Error on test set: {mse:.3f}")

# -----------------------
# Optional: Generate advice for each person in test set
# -----------------------
advice = {
    "Sleep Hours": "try to get more quality sleep each night 🤍",
    "Social Interaction Score": "spend more time connecting with loved ones; human connection is the best medicine of them all!",
    "Work Hours per Week": "find a better work-life balance; every aspect of your life matters! ✨",
    "Screen Time per Day (Hours)": "take some time off your screens — read a book, take a walk, or bake something 🌿",
    "Age": "age isn’t something to change — but experience brings wisdom 💗",
}

feature_importance = pd.Series(model.coef_, index=X_test_selected.columns)
meaningful_features = ["Age", "Sleep Hours", "Work Hours per Week",
                       "Screen Time per Day (Hours)", "Social Interaction Score"]
important_feats = feature_importance[meaningful_features].sort_values(ascending=False)

for i in range(len(X_test_selected)):
    sample = X_test_selected.iloc[[i]]
    pred_score = y_pred[i]
    low_feats = sample[important_feats.index].iloc[0].sort_values().index[:3]
    print(f"\nPerson {i+1} predicted happiness score: {pred_score:.2f}")
    print("🌈 To improve their happiness score, they could:")
    for feat in low_feats:
        msg = advice.get(feat, f"focus a bit more on {feat.lower()}")
        print(f" - {msg}")
