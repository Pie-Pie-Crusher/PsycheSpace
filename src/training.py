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
import random

# -----------------------
# Load dataset
# -----------------------
data = pd.read_csv("Data/trainingdata.csv")
print(f"Loaded dataset with {data.shape[0]} rows and {data.shape[1]} columns.")

# -----------------------
# Features and target
# -----------------------
X = data.drop("Happiness Score", axis=1)
y = data["Happiness Score"]

numeric_cols = ["Age", "Sleep Hours", "Work Hours per Week", "Screen Time per Day (Hours)", "Social Interaction Score"]

# Convert numeric columns to float
X[numeric_cols] = X[numeric_cols].astype(float)

# Encode categorical columns
X = pd.get_dummies(X, drop_first=True)

# Scale numeric features
scaler = MinMaxScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# -----------------------
# Feature selection
# -----------------------
selector = SelectKBest(score_func=f_regression, k='all')
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
X_selected = pd.DataFrame(X_selected, columns=selected_features)

print(f"Selected features: {list(selected_features)}")

# -----------------------
# Train-test split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# -----------------------
# Train model
# -----------------------
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

# -----------------------
# Friendly advice setup
# -----------------------
coefficients = model.coef_
feature_importance = pd.Series(coefficients, index=selected_features)
meaningful_features = ["Age", "Sleep Hours", "Work Hours per Week",
                       "Screen Time per Day (Hours)", "Social Interaction Score"]
meaningful_features = [f for f in meaningful_features if f in selected_features]
important_feats = feature_importance[meaningful_features].sort_values(ascending=False)

advice = {
    "Sleep Hours": "try to get more quality sleep each night 🤍",
    "Social Interaction Score": "spend more time connecting with loved ones; human connection is the best medicine of them all!",
    "Work Hours per Week": "find a better work-life balance; every aspect of your life matters! ✨",
    "Screen Time per Day (Hours)": "take some time off your screens — read a book, take a walk, or bake something 🌿",
    "Age": "age isn’t something to change — but experience brings wisdom 💗",
}

# -----------------------
# Random user prediction
# -----------------------
random_index = random.randint(0, len(X_selected) - 1)
user_df = X_selected.iloc[[random_index]]

predicted_happiness = model.predict(user_df)[0]
print(f"\nRandom user's predicted happiness score: {predicted_happiness:.2f}")

low_feats = user_df[important_feats.index].iloc[0].sort_values().index[:3]
print("🌈 To improve happiness score, they could:")
for feat in low_feats:
    msg = advice.get(feat, f"focus a bit more on {feat.lower()}")
    print(f" - {msg}")

print("\n---------------------------")
