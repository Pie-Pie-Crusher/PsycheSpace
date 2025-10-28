# app.py

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Import the trained model from models/model.py
from training.py import model

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

        # Extract input features in the correct order expected by your model
        features = [
            data["Country"],
            float(data["Age"]),
            data["Gender"],
            data["Exercise Level"],
            data["Diet Type"],
            float(data["Sleep Hours"]),
            data["Stress Level"],
            data["Mental Health Condition"],
            float(data["Work Hours per Week"]),
            float(data["Screen Time per Day (Hours)"]),
            float(data["Social Interaction Score"])
        ]

        # Make prediction
        happiness_score = model.predict([features])[0]

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
    app.run(debug=True)
