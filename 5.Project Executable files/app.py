# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input
from PIL import Image
import os

app = Flask(__name__)

# Load the SavedModel directory
model = load_model(r'C:\Users\rohan\OneDrive\Desktop\Plant Seedling Classification Project\seedling_exception.h5')

# Define the classes as in your Colab code
class_names = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Sugar beet', 'Shepherd’s purse', 'Small-flowered Cranesbill', 'Maize', 'Fennel', 'Oilseed rape', 'Common millet']

# Path to store uploaded images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')  # Route to display the home page
def home():
    return render_template('index.html')  # Rendering the home page

@app.route('/Prediction')
def prediction():
    return render_template('predict.html')

@app.route('/predict', methods=["POST", "GET"])
def upload():
    # Route to handle image upload and prediction
    if request.method == 'POST':
        # Get the uploaded file
        f = request.files['img']
        
        # Print current path for debugging
        print("Current path:")
        basepath = os.path.dirname(__file__)  # Correct usage of `__file__`
        print("Base path:", basepath)
        
        # Define the file path to save the uploaded file
        filepath = os.path.join(basepath, 'uploads', f.filename)
        print("Upload folder is:", filepath)
        
        # Save the file
        f.save(filepath)
        
        # Open and resize the image manually to (299, 299)
        img = Image.open(filepath)
        img = img.resize((299, 299))  # Manually resize to 299x299
        
        # Convert the image to a NumPy array
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)  # Add batch dimension
        
        # Preprocess the image for the model (apply model-specific preprocessing)
        x = preprocess_input(x)
        
        # Print the shape of the image to confirm resizing
        print("Shape of the image:", x.shape)
        
        # Predict using the model
        preds = np.argmax(model.predict(x), axis=1)
        
        # Map the prediction to class labels
        text = "The predicted seedling is: " + str(class_names[preds[0]])  # Ensure preds[0] is within bounds
        
        # Return the prediction result
        return render_template("predict.html", z=text)

    return render_template("upload.html")  # Render an upload page for GET requests

if __name__ == "__main__":
    app.run(debug=False)
