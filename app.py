# from flask import Flask, request, jsonify

# app = Flask(__name__)

# @app.route('/preprocess_image', methods=['POST'])
# def test_connectivity():
#   """
#   Endpoint to test API connectivity.
#   """
#   # No data processing needed, just respond with a success message.
#   return jsonify({"message": "successful"})

# if __name__ == '__main__':
# #   app.run(host='0.0.0.0', port=5000, debug=True)
#   app.run(debug=True, host='192.168.119.177', port=5000)


from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)

# Load VGG16 model for species detection
species_model = tf.keras.models.load_model('FishSpecie.h5')

# Load VGG16 model for disease detection
disease_model = tf.keras.models.load_model('DC.h5')

# Define class labels for species and disease detection
# species_labels = ['BettaFish', 'Gilt-Head Bream', 'GoldFish', 'GuppyFish', 'Molly', 'Red Mullet', 'Tetras']

species_labels = ['Bass','BettaFish','Black Sprat','Gilt-Head Bream','GoldFish','GuppyFish','Hourse Mackerel','Molly','Red Bream','Red Mullet','Tetras','Trout']
disease_labels = ['Healthy', 'EUS', 'ICH']


def preprocess_image(image_file):
    """Preprocesses an image for model prediction."""
    img = Image.open(image_file.stream)  # Open the image from the FileStorage object
    img = img.resize((224, 224))  # Resize the image to the desired dimensions
    img_array = np.array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array.astype(np.float32) / 255.0  # Normalize image data
    return img_array


@app.route('/process_image', methods=['POST'])
def process_image():
    # Check if image is present in request form data
    if 'image' not in request.files:
        return jsonify({'error': 'Missing image file'}), 400

    # Get the uploaded image file
    image_file = request.files['image']

    # Validate the image file (optional, adjust as needed)
    if image_file.filename.lower().endswith(('.png', '.jpg', '.jpeg', 'jfif')):
        # Securely save the image file (optional)
        image_path = secure_filename(image_file.filename)
        image_file.save(f'uploads/{image_path}')  # Save to a folder named 'uploads'
    else:
        return jsonify({'error': 'Invalid image format. Please upload a PNG, JPG, JFIF, or JPEG image'}), 400

    # Preprocess the image
    img_array = preprocess_image(image_file)

    # Predict species
    species_prediction = species_model.predict(img_array)
    species_index = np.argmax(species_prediction)
    species_result = species_labels[species_index]

    # Predict disease
    disease_prediction = disease_model.predict(img_array)
    disease_index = np.argmax(disease_prediction)
    disease_result = disease_labels[disease_index]

    # Return the results
    return jsonify({
        'species': species_result,
        'disease': disease_result
    })

if __name__ == '__main__':
    app.run(debug=True, host='192.168.119.177', port=5000)
