import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def scale_features(features):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features

def predict(model, scaled_features):
    pred_prob = model.predict(scaled_features)
    pred_labels = np.argmax(pred_prob, axis=1)
    return pred_labels

# scaling data for the student model

def scale_student_data(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

def predict_student(model, scaled_data):
    pred_gpa = model.predict(scaled_data)
    return pred_gpa