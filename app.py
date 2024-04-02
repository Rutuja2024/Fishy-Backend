from flask import Flask, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load VGG16 model for species detection
species_model = tf.keras.models.load_model('FC.h5')

# Load VGG16 model for disease detection
disease_model = tf.keras.models.load_model('DC.h5')

# Define class labels for species and disease detection
species_labels = ['BettaFish','Gilt-Head Bream','GoldFish','GuppyFish','Molly','Red Mullet','Tetras']  # Replace with your species labels
disease_labels = ['Healthy', 'EUS', 'ICH']  # Replace with your disease labels

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize image data
    return img_array

@app.route('/detect', methods=['GET'])
def detect():
    # Provide path to the image for testing
    img_path = './img3.jpg'  # Replace with the path to your image

    # Preprocess the image
    img_array = preprocess_image(img_path)

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
    app.run(debug=True)























