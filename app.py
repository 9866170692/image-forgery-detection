from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename

# Ensure the upload directory exists
UPLOAD_FOLDER = 'static/uploaded_images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model = tf.keras.models.load_model('model/cnn_model.h5')

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Handle file uploads
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part in the request."
    file = request.files['file']
    if file.filename == '':
        return "No selected file."
    if file:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess the image
        img = cv2.imread(filepath)
        img_resized = cv2.resize(img, (32, 32)) / 255.0
        img_expanded = np.expand_dims(img_resized, axis=0)

        # Predict using the model
        prediction = model.predict(img_expanded)
        predicted_label = np.argmax(prediction)

        # Determine classification
        classification = "Forged" if predicted_label == 1 else "Original"
        return render_template('index.html', classification=classification, image_path=filepath)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
