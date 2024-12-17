

import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Define the CNN Model Architecture
def create_model(input_shape=(224, 224, 3), num_classes=3):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))  # Output layer for classification
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Load and preprocess dataset
def load_data(dataset_path, img_size=(224, 224)):
    images = []
    labels = []
    class_names = os.listdir(dataset_path)
    for label, pose in enumerate(class_names):
        pose_folder = os.path.join(dataset_path, pose)
        for img_name in os.listdir(pose_folder):
            img_path = os.path.join(pose_folder, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, img_size)
            img = img / 255.0  # Normalize the image
            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels), class_names


# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


# Function to provide feedback based on detected pose
def give_feedback(predicted_pose):
    feedback_dict = {
        "Warrior II": {
            "left_knee": 90,
            "right_knee": 180,
        },
        "Tree Pose": {
            "left_knee": 90,
            "left_hip": 90,
        },
        "Downward Dog": {
            "left_knee": 180,
            "right_knee": 180,
            "back_angle": 45,  # Assuming we calculate back angle using landmarks
        },
        "Goddess Pose": {
            "left_knee": 90,
            "right_knee": 90,
            "hip_angle": 180,
            "back_angle": 90,
            "left_elbow": 90,
            "right_elbow": 90
        },
        "Plank Pose": {
            "shoulder_angle": 180,
            "hip_angle": 180,
            "elbow_angle": 180,
            "knee_angle": 180,
            "back_angle": 180
        }
        # Add feedback for other poses...
    }

    if predicted_pose in feedback_dict:
        ideal_angles = feedback_dict[predicted_pose]
        # Calculate the deviation and provide feedback (this part can use landmarks to calculate angles)
        print(f"Feedback: Align your joints to the ideal angles for {predicted_pose}: {ideal_angles}")
    else:
        print(f"No feedback available for {predicted_pose}")


# Function to predict yoga pose from an image
def predict_pose(img_path, model, class_names):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0  # Normalize image
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    prediction = model.predict(img)
    predicted_class_idx = np.argmax(prediction)

    predicted_pose = class_names[predicted_class_idx]
    return predicted_pose, prediction


# Load dataset and model
dataset_path = r'D:\yoga pse\pythonProject1\DATASET' # Path to your dataset
X, y, class_names = load_data(dataset_path)

# Convert labels to categorical (One Hot Encoding)
y = tf.keras.utils.to_categorical(y, num_classes=len(class_names))

# Create the model
model = create_model(num_classes=len(class_names))

# Train the model
model.fit(X, y, epochs=10, batch_size=32)

# Save the trained model
model.save('yoga_pose_model.h5')
# Load the saved model for evaluation
model = load_model('yoga_pose_model.h5')

# Load test data (ensure you have a separate test dataset for evaluation)
test_images, test_labels, _ = load_data(r'D:\yoga pse\TEST')  # Replace with actual test set path
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=len(class_names))

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_acc*100:.2f}%")

# Get predictions on the test set
predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)

# Show misclassified examples
for i in range(len(predicted_classes)):
    if predicted_classes[i] != np.argmax(test_labels[i]):
        print(f"Misclassified Image {i}: Actual - {CLASS_NAMES[np.argmax(test_labels[i])]}, Predicted - {CLASS_NAMES[predicted_classes[i]]}")

# Test prediction and feedback
test_image_path = r'D:\yoga pse\tree.jpg'  # Path to new test image
predicted_pose, prediction = predict_pose(test_image_path, model, class_names)
print(f"Predicted Pose: {predicted_pose}")
give_feedback(predicted_pose)

