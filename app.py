# app.py

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load("models/best_model.pkl")
print("✓ Model loaded successfully!")

# Create the Flask app
app = Flask(__name__, static_folder="website")
CORS(app)  # Allow JS from front-end to call /predict

# -----------------------------
# Serve the front-end
# -----------------------------
@app.route("/")
def home():
    """
    Serve the existing index.html file from the website folder.
    """
    return send_from_directory(app.static_folder, "index.html")

# -----------------------------
# Predict route
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON data for all features except Happiness Score:
    {
        "Country": "Brazil",
        "Age": 48,
        "Gender": "Male",
        "Exercise Level": "Low",
        "Diet Type": "Vegetarian",
        "Sleep Hours": 6.3,
        "Stress Level": "Low",
        "Mental Health Condition": "None",
        "Work Hours per Week": 21,
        "Screen Time per Day (Hours)": 4.0,
        "Social Interaction Score": 7.8
    }

    Returns JSON:
    {
        "prediction": 6.5
    }
    """
    try:
        data = request.get_json()
        print("\n=== NEW PREDICTION REQUEST ===")
        print(f"Received data: {data}")
        
        # Create encoding mappings (same as in preprocessing)
        exercise_encoding = {'Low': 1, 'Moderate': 2, 'High': 0}
        diet_encoding = {'Balanced': 0, 'Junk Food': 1, 'Keto': 2, 'Vegan': 3, 'Vegetarian': 4}
        stress_encoding = {'Low': 1, 'Moderate': 2, 'High': 0}
        mental_encoding = {'None': -0.5, 'Anxiety': 0, 'Bipolar': 1, 'Depression': 2, 'PTSD': 3}
        
        # Prepare features in the order the model expects
        # From your model: ['Age', 'Sleep Hours', 'Work Hours per Week', 'Screen Time per Day (Hours)', 
        #                  'Social Interaction Score', 'Exercise Level_encoded', 'Diet Type_encoded', 
        #                  'Stress Level_encoded', 'Mental Health Condition_encoded']
        
        features_dict = {
            'Age': float(data.get("Age", 30)),
            'Sleep Hours': float(data.get("Sleep Hours", 7)),
            'Work Hours per Week': float(data.get("Work Hours per Week", 40)),
            'Screen Time per Day (Hours)': float(data.get("Screen Time per Day (Hours)", 4)),
            'Social Interaction Score': float(data.get("Social Interaction Score", 6)),
            'Exercise Level_encoded': exercise_encoding.get(data.get("Exercise Level", "Moderate"), 2),
            'Diet Type_encoded': diet_encoding.get(data.get("Diet Type", "Balanced"), 0),
            'Stress Level_encoded': stress_encoding.get(data.get("Stress Level", "Low"), 1),
            'Mental Health Condition_encoded': mental_encoding.get(data.get("Mental Health Condition", "None"), -0.5)
        }
        
        # Convert to DataFrame for prediction
        input_df = pd.DataFrame([features_dict])
        print(f"Features prepared: {features_dict}")
        
        # Make prediction
        happiness_score = model.predict(input_df)[0]
        print(f"Raw prediction: {happiness_score}")
        
        # Ensure score is between 0 and 10
        happiness_score = max(0, min(10, float(happiness_score)))
        print(f"Final happiness score: {happiness_score}")
        print("=== PREDICTION COMPLETE ===\n")

        return jsonify({"prediction": happiness_score})

    except KeyError as e:
        return jsonify({"error": f"Missing input feature: {e}"}), 400
    except ValueError as e:
        return jsonify({"error": f"Invalid data type: {e}"}), 400
    except Exception as e:
        return jsonify({"error": f"Error making prediction: {e}"}), 400

# -----------------------------
# Run the Flask app
# -----------------------------
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
