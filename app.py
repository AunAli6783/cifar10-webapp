from flask import Flask, render_template, request, jsonify
from tensorflow import keras
import numpy as np
from PIL import Image
import os
import tensorflow as tf

app = Flask(__name__)

# CIFAR-10 class names
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# List of possible model files to try
MODEL_FILES = [
    'compatible_model.h5',
    'cifar_model.h5',
    'cifar_model_compatible.h5',
    'cifar_model_improved.h5'
]

def try_load_model(model_path):
    try:
        print(f"Attempting to load model from: {model_path}")
        model = tf.keras.models.load_model(model_path, compile=False)
        print(f"Successfully loaded model from: {model_path}")
        return model
    except Exception as e:
        print(f"Error loading {model_path}: {str(e)}")
        return None

# Try loading models in sequence
model = None
for model_file in MODEL_FILES:
    if os.path.exists(model_file):
        model = try_load_model(model_file)
        if model is not None:
            print(f"Using model from: {model_file}")
            break

if model is None:
    print("Failed to load any model file!")

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
    if model is None:
        return jsonify({'error': 'Model not loaded properly. Please check the server logs.'})
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    try:
        img = Image.open(file.stream)
        img_array = prepare_image(img)
        preds = model.predict(img_array, verbose=0)[0]
        
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
