from preprocessing import load_data, preprocess_data, split_data, scale_data
from model import build_model, train_model
from prediction import predict
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
