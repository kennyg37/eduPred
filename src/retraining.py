from src.preprocessing import load_data, preprocess_data, process_student_data, split_data, scale_data, split_student_data
from src.model import build_model, build_student_model, train_model, train_student_model
from src.prediction import predict, predict_student, scale_student_data
from tensorflow.keras.utils import to_categorical # type: ignore

def retrain_model(data_path):
    df = load_data(data_path)
    X, y = preprocess_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    X_train, X_val, X_test = scale_data(X_train, X_val, X_test)
    
    y_train_encoded = to_categorical(y_train, num_classes=3)
    y_val_encoded = to_categorical(y_val, num_classes=3)
    y_test_encoded = to_categorical(y_test, num_classes=3)
    
    model = build_model(X_train.shape[1])
    model, history = train_model(model, X_train, y_train_encoded, X_val, y_val_encoded)
    pred_labels = predict(model, X_test)
    return model

def retrain_student_model(data_path):
    df = load_data(data_path)
    X, y = process_student_data(df)
    X_scaled = scale_student_data(X)
    X_train, X_val, X_test, y_train, y_val, y_test = split_student_data(X_scaled, y)
    
    model = build_student_model(X_train.shape[1])
    model, history = train_student_model(model, X_train, y_train, X_val, y_val)
    pred_gpa = predict_student(model, X_test)
    return model
