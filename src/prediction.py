import numpy as np

def predict(model, X_test):
    pred_prob = model.predict(X_test)
    pred_labels = np.argmax(pred_prob, axis=1)
    return pred_labels
