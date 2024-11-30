import os
from flask import Flask, jsonify, redirect, request, send_from_directory
from dotenv import load_dotenv
import numpy as np
from src.prediction import predict, scale_features, scale_student_data
from src.retraining import retrain_model, retrain_student_model
from flask_swagger_ui import get_swaggerui_blueprint
from werkzeug.utils import secure_filename

load_dotenv()

app = Flask(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
port = int(os.getenv("PORT", 5000))
API_URL = os.getenv("API_URL", '/static/swagger.json')

UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", '/tmp/uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

swaggerui_bluprint = get_swaggerui_blueprint('/docs', API_URL)
app.register_blueprint(swaggerui_bluprint)

STUDENT_TEMP_MODEL_PATH = os.getenv("STUDENT_TEMP_MODEL_PATH", '/tmp/student_gpa_model.keras')
ED_TEMP_MODEL_PATH = os.getenv("ED_TEMP_MODEL_PATH", '/tmp/edupred.keras')

MODEL_PATH = os.getenv("MODEL_PATH", './model/edupred.keras')
model = None
STUDENT_MODEL_PATH = os.getenv("STUDENT_MODEL_PATH", './model/student_model.keras')
student_model = None

@app.route('/', methods=['GET'])
def home():
    return redirect('/docs')

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
        # Retrain the mode using the retrain_model funtion
        new_model = retrain_model()
        new_model.save(STUDENT_TEMP_MODEL_PATH)
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

    expected_features = ['Age', 'Gender', 'Ethnicity', 'Parental Education', 'Weekly study time', 'Absences', 'Tutoring', 'Parental support', 'extracullicular', 'sports', 'music', 'volunteering']

    data = request.json.get('features', None)
    if data is None:
        return jsonify({"error": "No features provided for prediction"}), 400

    missing_keys = [key for key in expected_features if key not in data]
    if missing_keys:
        return jsonify({"error": f"Missing keys: {missing_keys}"}), 400
    try:
        feature_array = [data[key] for key in expected_features]
        feature_array = np.array(feature_array).reshape(1, -1)
        scaled_features = scale_student_data(feature_array)
        predictions = predict(student_model, scaled_features)
        predictions = [int(pred) if isinstance(pred, np.integer) else float(pred) for pred in predictions]
        return jsonify({"predicted_gpa": predictions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/student/retrain', methods=['POST'])
def retrain_student_endpoint():
    """Endpoint to retrain the model."""
    global student_model
    try:
        file = request.files.get('file')
        
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
        else:
            file_path = './data/Student_performance_data _.csv'
        # Retrain the mode using the retrain_model function
        new_model = retrain_student_model(file_path)
        new_model.save(STUDENT_MODEL_PATH)
        student_model = new_model
        return jsonify({"message": "Model retrained successfully."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=port, debug=True)