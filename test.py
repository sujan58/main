import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers import Adam
from PIL import UnidentifiedImageError
from collections import Counter
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from data_preprocess import X_test,X_train,test_datagen,train_datagen,y_test,y_train,load_images_from_folders,recyclable_folder,non_recyclable_folder
from model_build import model
from train import history


# Model Evaluation
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test Accuracy:', test_acc)

# Save the model
model.save('model_save/recyclable_classifier_model.h5')



# Plotting the training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plotting the training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Load and preprocess the sample image
sample_image_path = 'dataset/recyclable/fabric_9.png'
sample_image = cv2.imread(sample_image_path)
sample_image = cv2.resize(sample_image, (200, 200))  # Resize image to match model input size
sample_image = np.expand_dims(sample_image, axis=0)  # Add batch dimension

# Make prediction
prediction = model.predict(sample_image)

# Decode prediction (assuming binary classification)
if prediction[0][0] > 0.5:
    print("Predicted: Recyclable")
else:
    print("Predicted: Non-recyclable")


    

