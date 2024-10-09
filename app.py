from flask import Flask, request, jsonify
from flask_cors import CORS  
import tensorflow as tf
import numpy as np
from PIL import Image
import io


app = Flask(__name__)
CORS(app)  


model = tf.keras.models.load_model('my_mnist_model.keras')  


def preprocess_image(image):
    
    image = image.resize((28, 28)).convert('L')

   
    image_array = np.array(image)

    
    image_array = image_array.reshape(1, 28, 28, 1)

    
    image_array = image_array.astype('float32') / 255.0

    return image_array

@app.route('/', methods=['GET'])
def index():
    return 'Flask app is running'   


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    
    try:
        image = Image.open(io.BytesIO(file.read()))
        
        preprocessed_image = preprocess_image(image)

        
        prediction = model.predict(preprocessed_image)

        
        predicted_class = np.argmax(prediction, axis=1)[0]

        return jsonify({"predicted_class": int(predicted_class)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
