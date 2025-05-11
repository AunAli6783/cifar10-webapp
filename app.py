from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# CIFAR-10 class names
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Load the trained model
MODEL_PATH = 'cifar_model_improved.h5'
model = load_model(MODEL_PATH)

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
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    try:
        img = Image.open(file.stream)
        img_array = prepare_image(img)
        preds = model.predict(img_array)[0]
        
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