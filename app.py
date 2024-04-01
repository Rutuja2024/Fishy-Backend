# from flask import Flask, jsonify
# import torch
# from torchvision import transforms
# from PIL import Image
# import numpy as np
# import os

# app = Flask(__name__)

# # Load YOLOv5 model for object detection
# yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='last.pt')

# # Load VGG16 model for species detection
# species_model = tf.keras.models.load_model('FC.h5')

# # Load VGG16 model for disease detection
# disease_model = tf.keras.models.load_model('DC.h5')

# # Define class labels for species and disease detection
# species_labels = ['BettaFish','Gilt-Head Bream','GoldFish','GuppyFish','Molly','Red Mullet','Tetras']  # Replace with your species labels
# disease_labels = ['Healthy', 'RedSpot', 'WhiteSpot']  # Replace with your disease labels

# def preprocess_yolov5_image(img_path):
#     img = Image.open(str(img_path)).convert('RGB')  # Convert to string before using
#     transform = transforms.Compose([
#         transforms.Resize(640),  # Resize to 640x640 for YOLOv5
#         transforms.ToTensor(),
#     ])
#     img_array = transform(img).unsqueeze(0)
#     return img_array

# def preprocess_vgg16_image(img_path):
#     img = Image.open(str(img_path)).convert('RGB')  # Convert to string before using
#     transform = transforms.Compose([
#         transforms.Resize(224),  # Resize to 224x224 for VGG16
#         transforms.ToTensor(),
#     ])
#     img_array = transform(img).unsqueeze(0)
#     return img_array

# @app.route('/detect', methods=['GET'])
# def detect():
#     # Provide path to the image for testing
#     img_path = './img1.jpg'  # Replace with the path to your image

#     # Preprocess the image for YOLOv5
#     yolo_img_array = preprocess_yolov5_image(img_path)

#     # Perform object detection and annotation using YOLOv5
#     annotated_image = yolo_model(yolo_img_array)

#     # Preprocess the image for VGG16
#     vgg16_img_array = preprocess_vgg16_image(img_path)

#     # Perform species detection using VGG16
#     species_prediction = species_model.predict(vgg16_img_array)
#     species_index = np.argmax(species_prediction)
#     species_result = species_labels[species_index]

#     # Perform disease detection using VGG16
#     disease_prediction = disease_model.predict(vgg16_img_array)
#     disease_index = np.argmax(disease_prediction)
#     disease_result = disease_labels[disease_index]

#     # Return the annotated image and detection results
#     return jsonify({
#         'annotated_image': annotated_image,  # You may need to convert annotated_image to a suitable format for JSON serialization
#         'species': species_result,
#         'disease': disease_result
#     })

# if __name__ == '__main__':
#     app.run(debug=True)



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
disease_labels = ['Healthy', 'RedSpot', 'WhiteSpot']  # Replace with your disease labels

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize image data
    return img_array

@app.route('/detect', methods=['GET'])
def detect():
    # Provide path to the image for testing
    img_path = './img1.jpg'  # Replace with the path to your image

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























