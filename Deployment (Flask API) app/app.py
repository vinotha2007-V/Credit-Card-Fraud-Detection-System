from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("../models/fraud_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']
    prediction = model.predict([data])
    
    return jsonify({
        "fraud": int(prediction[0])
    })

if __name__ == '__main__':
    app.run(debug=True)
