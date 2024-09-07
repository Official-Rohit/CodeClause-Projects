from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model

# Initialize Flask apps
app = Flask(__name__)

# Load the Keras model
model = load_model(r'E:\CU\Year 2\Projects\CLVP - CC\new_directory_name\credit_default_model.keras')  # Keras model loading

@app.route('/')
def home():
    return "ML Model API is Running"

# Define a predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data sent via POST request
        data = request.json
        input_data = np.array(data['input']).reshape(1, -1)  # Adjust for your model's input shape
        
        # Make predictions using the model
        prediction = model.predict(input_data)
        
        # Send the prediction back as a response
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
