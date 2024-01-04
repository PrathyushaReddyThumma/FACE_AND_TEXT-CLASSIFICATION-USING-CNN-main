import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Define the dataset directory and emotion labels
data_dir = "path/to/dataset"
emotion_labels = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Define a function to preprocess the images
def preprocess_image(image):
    # Resize the image to 48x48 pixels
    image = cv2.resize(image, (48, 48))
    # Convert the image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Normalize the pixel values to be between 0 and 1
    image = image / 255.0
    # Add a channel dimension to the image
    image = np.expand_dims(image, axis=-1)
    return image

# Define a function to read the dataset
def read_dataset():
    # Initialize empty lists to store the images and labels
    images = []
    labels = []
    # Loop through the emotion labels
    for i, label in enumerate(emotion_labels):
        # Define the label directory
        label_dir = os.path.join(data_dir, label)
        # Loop through the images in the label directory
        for filename in os.listdir(label_dir):
            # Read the image from file
            image = cv2.imread(os.path.join(label_dir, filename))
            # Preprocess the image
            image = preprocess_image(image)
            # Add the image and label to the lists
            images.append(image)
            labels.append(i) # use integer labels instead of one-hot encoding
    # Convert the lists to NumPy arrays
    images = np.array(images)
    labels = np.array(labels)
    # Split the dataset into training and test sets
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
    # Split the training set into training and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)
    # Return the preprocessed dataset
    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)

# Read the dataset
(train_images, train_labels), (val_images, val_labels), (test_images, test_labels) = read_dataset()

# Print the shapes of the dataset
print("Training set shape:", train_images.shape, train_labels.shape)
print("Validation set shape:", val_images.shape, val_labels.shape)
print("Test set shape:", test_images.shape, test_labels.shape)
