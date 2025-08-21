from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model and encoder
with open('decision_tree_model.pkl', 'rb') as file:
    model, encoder_sodium = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = int(request.form['age'])
        blood_pressure = int(request.form['blood_pressure'])
        diet = request.form['diet']

        diet_encoded = 0 if diet == 'healthy' else 1
        input_data = np.array([[age, blood_pressure, diet_encoded]])

        # Predict (returns a number)
        prediction_encoded = model.predict(input_data)[0]

        # Convert numeric prediction back to label
        predicted_sodium = encoder_sodium.inverse_transform([prediction_encoded])[0]

        return jsonify({'prediction': predicted_sodium})
    
    except Exception as e:
        return jsonify({'error': str(e)})

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

