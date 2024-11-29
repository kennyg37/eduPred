import os
from flask import Flask, jsonify, request
from dotenv import load_dotenv
import numpy as np
from src.prediction import predict, scale_features, scale_student_data
# from src.preprocessing import scale_data
from src.retraining import retrain_model, retrain_student_model

load_dotenv()

app = Flask(__name__)

port = int(os.getenv("PORT", 5000))

# Load the model when the app starts
MODEL_PATH = './model/edupred.keras'
model = None
STUDENT_MODEL_PATH = './model/student_model.keras'
student_model = None

if os.path.exists(MODEL_PATH):
    from tensorflow.keras.models import load_model # type: ignore
    model = load_model(MODEL_PATH)

@app.route('/edupred/predict', methods=['POST'])
def predict_endpoint():
    """Endpoint for making predictions."""
    global model
    if model is None:
        return jsonify({"error": "Model not loaded. Retrain the model first."}), 500

    data = request.json.get('features', None)
    if data is None:
        return jsonify({"error": "No features provided for prediction"}), 400

    try:
        features = np.array(data).reshape(1, -1) 
        scaled_features = scale_features(features)
        predictions = predict(model, scaled_features)
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/edupred/retrain', methods=['POST'])
def retrain_endpoint():
    """Endpoint to retrain the model."""
    global model
    try:
        # Retrain the model using the retrain_model function
        new_model = retrain_model()
        new_model.save(MODEL_PATH)
        model = new_model
        return jsonify({"message": "Model retrained successfully."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if os.path.exists(STUDENT_MODEL_PATH):
    from tensorflow.keras.models import load_model # type: ignore
    student_model = load_model(STUDENT_MODEL_PATH)

@app.route('/student/predict', methods=['POST'])
def predict_student_endpoint():
    """Endpoint for making predictions."""
    global student_model
    if student_model is None:
        return jsonify({"error": "Model not loaded. Retrain the model first."}), 500

    data = request.json.get('features', None)
    if data is None:
        return jsonify({"error": "No features provided for prediction"}), 400

    try:
        data = np.array(data).reshape(1, -1) 
        scaled_features = scale_student_data(data)
        predictions = predict(student_model, scaled_features)
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/student/retrain', methods=['POST'])
def retrain_student_endpoint():
    """Endpoint to retrain the model."""
    global student_model
    try:
        # Retrain the model using the retrain_model function
        new_model = retrain_student_model()
        new_model.save(STUDENT_MODEL_PATH)
        student_model = new_model
        return jsonify({"message": "Model retrained successfully."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=port, debug=True)
    