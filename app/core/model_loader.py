import joblib
import numpy as np

model = joblib.load("model/model.pkl")

def predict_with_model(features):
    arr = np.array(features).reshape(1,-1)
    return int(model.predict(arr)[0])