from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.saving import register_keras_serializable
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import io
import os

app = Flask(__name__)

# Custom functions
@register_keras_serializable()
def depth_to_space(x):
    return tf.nn.depth_to_space(x, block_size=2)

def psnr(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))

def ssim(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

# Load models
MODEL_256_PATH = 'edsr_model_256_to_512.h5'
MODEL_512_PATH = 'edsr_model_512_to_1024.h5'

model_256 = load_model(MODEL_256_PATH, custom_objects={'psnr': psnr, 'ssim': ssim, 'depth_to_space': depth_to_space})
model_512 = load_model(MODEL_512_PATH, custom_objects={'psnr': psnr, 'ssim': ssim, 'depth_to_space': depth_to_space})

# Image processing
def preprocess_image(image, size):
    image = image.convert('L')  # grayscale
    image = image.resize(size, Image.LANCZOS)
    arr = np.array(image) / 255.0
    return np.expand_dims(np.expand_dims(arr, axis=0), axis=-1)

def postprocess_image(array):
    array = np.clip(array * 255.0, 0, 255).astype('uint8')
    return Image.fromarray(array.squeeze())


@app.route('/enhance', methods=['POST'])
def enhance():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided', 'success': False}), 400

        file = request.files['image']
        image = Image.open(file.stream)
        original_size = image.size
        max_dim = max(original_size)

        if max_dim < 350:
            model = model_256
            input_size = (256, 256)
            output_size = '512x512'
        else:
            model = model_512
            input_size = (512, 512)
            output_size = '1024x1024'

        processed = preprocess_image(image, input_size)
        enhanced = model.predict(processed)
        enhanced_img = postprocess_image(enhanced)

        buf = io.BytesIO()
        enhanced_img.save(buf, format='JPEG')
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')

        return jsonify({
            'success': True,
            'enhanced_image': img_str,
            'original_size': f'{original_size[0]}x{original_size[1]}',
            'enhanced_size': output_size
        })

    except Exception as e:
        import traceback
        traceback.print_exc()  # <-- PRINT THE FULL ERROR STACK
        return jsonify({'error': str(e), 'success': False}), 500


app = Flask(__name__)
    
@app.route("/")
def home():
    return render_template("index.html")


