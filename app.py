import os
from flask import Flask, jsonify, request
from dotenv import load_dotenv
import numpy as np
from prediction import predict
from preprocessing import scale_data
from retraining import retrain_model

load_dotenv()

app = Flask(__name__)

port = int(os.getenv("PORT", 5000))

# Load the model when the app starts
MODEL_PATH = './models/model_weights.h5'
model = None

if os.path.exists(MODEL_PATH):
    from tensorflow.keras.models import load_model # type: ignore
    model = load_model(MODEL_PATH)

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """Endpoint for making predictions."""
    global model
    if model is None:
        return jsonify({"error": "Model not loaded. Retrain the model first."}), 500

    data = request.json.get('features', None)
    if data is None:
        return jsonify({"error": "No features provided for prediction"}), 400

    try:
        features = np.array(data).reshape(1, -1)  # Ensure data is in the right shape
        scaled_features = scale_data(features)   # Scale input
        predictions = predict(model, scaled_features)
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/retrain', methods=['POST'])
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

if __name__ == "__main__":
    app.run(port=port, debug=True)
    