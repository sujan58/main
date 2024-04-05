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
from data_preprocess import X_test, X_train, test_datagen, train_datagen, y_test, y_train, load_images_from_folders, recyclable_folder, non_recyclable_folder
from model_build import model

# Model Training
history = model.fit(train_datagen.flow(X_train, y_train, batch_size=32),
                    steps_per_epoch=len(X_train) // 32,
                    epochs=50,
                    validation_data=test_datagen.flow(X_test, y_test, batch_size=32),
                    validation_steps=len(X_test) // 32)

# Error handling for NaN values in pie chart data
sizes = [10, 20, 30, np.nan]  # Example data with NaN value
labels = ['A', 'B', 'C', 'D']

# Remove NaN values and corresponding labels
data = zip(sizes, labels)
data_cleaned = [(size, label) for size, label in data if not np.isnan(size)]
sizes_cleaned, labels_cleaned = zip(*data_cleaned)

# Check if sizes list is empty after removing NaN values
if not sizes_cleaned:
    print("Error: Sizes list is empty.")
else:
    # Calculate pie chart
    plt.pie(sizes_cleaned, labels=labels_cleaned, autopct='%1.1f%%', startangle=140)
    plt.show()
