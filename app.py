import os
from flask import Flask, request, render_template, url_for
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained yoga pose detection model
try:
    model = load_model('yoga_pose_model.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Class names corresponding to model outputs
CLASS_NAMES = ["Downward Dog", "Goddess Pose", "Plank Pose", "Tree Pose", "Warrior II"]

# Initialize Flask app
app = Flask(__name__)

# Configuration for upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to validate file type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            print("No file part in the request.")
            return "Error: No file uploaded.", 400

        file = request.files['file']

        if file.filename == '':
            print("No file selected.")
            return "Error: No file selected.", 400

        if not allowed_file(file.filename):
            print("Invalid file format.")
            return "Error: Allowed formats are PNG, JPG, JPEG.", 400

        # Save the file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        print(f"File saved at {file_path}")

        # Predict the pose
        predictions, pose_name = predict_pose(file_path)
        print(f"Pose predicted: {pose_name}")

        # Generate static file URL
        file_url = url_for('static', filename=f'uploads/{file.filename}')
        return render_template('result.html', pose=pose_name, file_path=file_url)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return "An error occurred. Check the logs for details.", 500

# Function to predict pose
def predict_pose(file_path):
    try:
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError("Image could not be loaded. Check the file format.")
        print(f"Image loaded with shape: {img.shape}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        predictions = model.predict(img)
        predicted_class_idx = np.argmax(predictions)
        pose_name = CLASS_NAMES[predicted_class_idx]
        return predictions, pose_name
    except Exception as e:
        print(f"Error in pose prediction: {e}")
        raise

if __name__ == '__main__':
    app.run(debug=True)
