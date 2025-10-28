#!/usr/bin/env python3
"""
model.py - Train a model to predict happiness scores
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# Load the processed dataset
data = pd.read_csv("../Data/processed_data.csv")
print(f"Loaded processed dataset with {data.shape[0]} rows and {data.shape[1]} columns.")

# Define features and target
# Exclude Country and Gender as requested
feature_cols = ['Age', 'Sleep Hours', 'Work Hours per Week', 'Screen Time per Day (Hours)', 
                'Social Interaction Score', 'Exercise Level_encoded', 'Diet Type_encoded', 
                'Stress Level_encoded', 'Mental Health Condition_encoded']
target_col = 'Happiness Score'

X = data[feature_cols]
y = data[target_col]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# Train multiple models
models = {
    'Ridge': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

best_model = None
best_score = -float('inf')

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"\n{name}:")
    print(f"  R² score: {r2:.3f}")
    print(f"  MSE: {mse:.3f}")
    
    if r2 > best_score:
        best_score = r2
        best_model = model
        best_model_name = name

print(f"\nBest model: {best_model_name} with R² score: {best_score:.3f}")

# Save the best model
joblib.dump(best_model, "best_model.pkl")
print("Best model saved to 'best_model.pkl'")
