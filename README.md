# yogapose_detect
Overview
The Yoga Pose Detection project classifies yoga poses from images into predefined categories using deep learning. This system leverages computer vision and a Convolutional Neural Network (CNN) to identify poses. It also features a user-friendly web interface that enables real-time predictions.

# Project Workflow
The project follows a structured methodology, from data preparation to deployment.

 1. Data Collection
Images of various yoga poses were collected from publicly available datasets and manually curated to include the following classes:
Warrior II
Tree Pose
Downward Dog
Goddess Pose
Plank Pose

2. Data Preprocessing
To ensure consistency and improve model performance, several preprocessing steps were applied:
Image Resizing: Resized all images to 224x224 pixels to match the CNN input requirements.
Normalization: Scaled pixel values to the range [0, 1].
Data Augmentation: Enhanced dataset diversity and reduced overfitting using:
Random rotations
Horizontal and vertical flips
Zoom and brightness adjustments

 3. Model Architecture
A custom CNN was designed for this project, optimized for multi-class image classification tasks.
Model Highlights:
Input Layer: Accepts images of size 224x224x3.
Convolutional Layers: Extract features through multiple filters.
Batch Normalization: Stabilizes training by normalizing intermediate layers.
Pooling Layers: Reduces spatial dimensions to improve computational efficiency.
Fully Connected Layers: Combines features for classification.

Output Layer:
Contains five neurons (one for each pose).
Utilizes the softmax activation function for multi-class predictions.
Training Parameters:
Loss Function: Categorical Crossentropy
Optimizer: Adam optimizer for faster convergence

#4. Web Application
An interactive web interface was developed for seamless user experience.
Backend: Flask handles image uploads and runs the prediction pipeline.
Frontend: Provides a clean and responsive UI for:
Uploading images
Displaying classification results dynamically
Integration: The trained CNN model is integrated into the Flask app for real-time inference.

 Results
The model demonstrates robust performance:
 Training Accuracy: ~96%
 Validation Accuracy: ~92%
 Test Accuracy: ~90% on unseen data

# Acknowledgments
Inspiration: This project draws inspiration from fitness apps and advancements in computer vision.
Dataset: Special thanks to publicly available yoga datasets, such as the Yoga Poses Dataset.

# Future Enhancements
Model Improvements
Experiment with transfer learning using pre-trained models like ResNet or EfficientNet.
Expand the dataset to enhance generalization capabilities.
Real-Time Detection
Integrate OpenCV or Mediapipe for real-time pose detection through webcams.
Web UI Enhancements
Provide interactive feedback for incorrect poses.
Offer posture improvement tips based on predictions.
Mobile Deployment
Optimize the model for TensorFlow Lite to enable mobile usage.
