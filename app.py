from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import os
import onnxruntime as ort

app = Flask(__name__)

# CIFAR-10 class names
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Load the ONNX model
MODEL_PATH = 'compatible_model.onnx'
try:
    session = ort.InferenceSession(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    session = None

def prepare_image(img):
    # Ensure image is 32x32 and RGB
    img = img.resize((32, 32))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if session is None:
        return jsonify({'error': 'Model not loaded properly. Please check the server logs.'})
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    try:
        img = Image.open(file.stream)
        img_array = prepare_image(img)
        
        # Get the input name from the model
        input_name = session.get_inputs()[0].name
        
        # Run inference
        preds = session.run(None, {input_name: img_array})[0][0]
        
        # Get top 3 predictions
        top_3_idx = np.argsort(preds)[-3:][::-1]  # Get indices of top 3 predictions
        predictions = [
            {
                'class': CLASS_NAMES[idx],
                'confidence': float(preds[idx])
            }
            for idx in top_3_idx
        ]
        
        return jsonify({'predictions': predictions})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
