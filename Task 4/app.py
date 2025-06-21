# app.py
from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import joblib
import base64

app = Flask(__name__)
model = joblib.load("model/gesture_model.pkl")

CLASS_NAMES = [
    "Palm", "L‑shape", "Fist", "Fist‑moved", "Index", 
    "Ok‑sign", "Palm‑moved", "C‑shape", "Down", "Unknown"
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data     = request.json['image'].split(',')[1]
    nparr    = np.frombuffer(base64.b64decode(data), np.uint8)
    img      = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    img      = cv2.resize(img, (64, 64)).flatten().reshape(1, -1) / 255.0
    pred_idx = model.predict(img)[0]
    gesture  = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else "Unknown"
    return jsonify({ 'prediction_idx': int(pred_idx), 'prediction_name': gesture })

if __name__ == '__main__':
    app.run(debug=True)

