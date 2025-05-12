import tensorflow as tf
import tf2onnx
import onnx

# Load the TensorFlow model
model = tf.keras.models.load_model('compatible_model.h5', compile=False)

# Convert to ONNX
input_signature = [tf.TensorSpec([1, 32, 32, 3], tf.float32, name="input")]
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)

# Save the ONNX model
onnx.save(onnx_model, 'compatible_model.onnx')
print("Model converted and saved as compatible_model.onnx") 