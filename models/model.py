# app.py

# Import necessary libraries
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle  # or use import joblib if your model is a .joblib file
import os

# Create Flask app and enable CORS so front-end JS can call /predict
app = Flask(__name__, static_folder="website")
CORS(app)

# ----------------------------
# Load the trained model
# ----------------------------
# Make sure to replace 'model.pkl' with your actual model file name if different
MODEL_PATH = "models/model.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ----------------------------
# Route: Serve the front-end
# ----------------------------
@app.route("/")
def home():
    """
    Serve the existing index.html from the static folder.
    """
    return send_from_directory(app.static_folder, "index.html")

# ----------------------------
# Route: Predict
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON input with features in the correct order.
    Example request JSON:
    {
      "subscription_type": "free",
      "listening_time": 42,
      "songs_played_per_day": 18,
      "skip_rate": 0.22,
      "ads_listened_per_week": 12
    }

    Returns JSON prediction:
    {
      "prediction": "will churn"
    }
    """
    try:
        data = request.get_json()  # Parse JSON from request body

        # Replace these feature names with your model's required input features
        required_features = [
            "subscription_type",
            "listening_time",
            "songs_played_per_day",
            "skip_rate",
            "ads_listened_per_week"
        ]

        # Extract features in order
        input_values = []
        for feature in required_features:
            if feature not in data:
                return jsonify({"error": f"Missing feature: {feature}"}), 400
            input_values.append(data[feature])

        # If your model expects a 2D array for a single prediction
        prediction = model.predict([input_values])[0]

        # Return prediction as JSON
        return jsonify({"prediction": str(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ----------------------------
# Run the app
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)
