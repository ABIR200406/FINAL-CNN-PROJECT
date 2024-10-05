from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('trained_model.h5')

# CIFAR-10 class names
class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

@app.route('/')
def home():
    return "Welcome to the CIFAR Image Classification API!"

# Route to handle image prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Read the image file and preprocess it
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    
    if img is None:
        return jsonify({'error': 'Invalid image file'})

    # Resize the image to 32x32 (CIFAR-10 image size)
    img = cv2.resize(img, (32, 32))
    
    # Convert image to array and normalize it
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for batch input
    img_array = img_array / 255.0  # Normalize

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])

    # Return the result as JSON
    return jsonify({
        'predicted_class': class_names[predicted_class],
        'confidence': str(np.max(predictions[0]))
    })

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
