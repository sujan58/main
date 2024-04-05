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


data_dir = r'C:\Users\dhanu\Downloads\WasteDevelopment\dataset'


recyclable_folder = os.path.join(data_dir, 'recyclable')
non_recyclable_folder = os.path.join(data_dir, 'non-recyclable')

print(recyclable_folder,non_recyclable_folder)

#Exploring data

def load_images_from_folder(folder):
    images = []
    for subdir, _, files in os.walk(folder):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                img = cv2.imread(os.path.join(subdir, file))
                if img is not None:
                    images.append(img)
    return images



recyclable_images = load_images_from_folder('dataset/recyclable')
non_recyclable_images = load_images_from_folder('dataset/non_recyclable')

total_recyclable_images = len(recyclable_images)
total_non_recyclable_images = len(non_recyclable_images)

print("Total Recyclable Images:", total_recyclable_images)
print("Total Non-Recyclable Images:", total_non_recyclable_images)

labels = ["Recyclable","Non-Recyclable"]
sizes = [total_recyclable_images, total_non_recyclable_images]


plt.figure(figsize=(8, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.title('Distribution of Images')
plt.show()

# Visualize sample images
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i in range(5):
    axes[0, i].imshow(cv2.cvtColor(recyclable_images[i], cv2.COLOR_BGR2RGB))
    axes[0, i].set_title('Recyclable')
    axes[0, i].axis('off')
    axes[1, i].imshow(cv2.cvtColor(non_recyclable_images[i], cv2.COLOR_BGR2RGB))
    axes[1, i].set_title('Non-Recyclable')
    axes[1, i].axis('off')
plt.show()

# Analyze image dimensions
recyclable_dimensions = [img.shape[:2] for img in recyclable_images]
non_recyclable_dimensions = [img.shape[:2] for img in non_recyclable_images]

recyclable_counter = Counter(recyclable_dimensions)
non_recyclable_counter = Counter(non_recyclable_dimensions)

print("Recyclable Image Dimensions:")
print(recyclable_counter)
print("Non-Recyclable Image Dimensions:")
print(non_recyclable_counter)

def load_images_from_folders(folder):
    images = []
    labels = []
    for subdir, _, files in os.walk(folder):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                label = os.path.basename(subdir)
                img = cv2.imread(os.path.join(subdir, file))
                if img is not None:
                    img = cv2.resize(img, (200, 200)) # Resize images to a uniform size
                    images.append(img)
                    labels.append(label)
    return np.array(images), np.array(labels)

recyclable_images, recyclable_labels = load_images_from_folders('dataset/recyclable')

non_recyclable_images, non_recyclable_labels = load_images_from_folders('dataset/non_recyclable')


def resize_images_from_folder(folder, target_size=(200, 200)):
    images = []
    for subdir, _, files in os.walk(folder):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(subdir, file)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, target_size)
                    images.append(img)
    return np.array(images)


recyclable_images = resize_images_from_folder('dataset/recyclable')
non_recyclable_images = resize_images_from_folder('dataset/non_recyclable')

print("Recyclable Image Shapes:", Counter([img.shape[:2] for img in recyclable_images]))
print("Non-Recyclable Image Shapes:", Counter([img.shape[:2] for img in non_recyclable_images]))

# Combine recyclable and non-recyclable data
all_images = np.concatenate((recyclable_images, non_recyclable_images), axis=0)
all_labels = np.concatenate((recyclable_labels, non_recyclable_labels), axis=0)


print(all_labels)

# One-hot encoding labels
all_labels = np.where(all_labels == 'recyclable', 1, 0)

X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

len(X_train), len(X_test)
# Data Augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)




train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
test_generator = test_datagen.flow(X_test, y_test, batch_size=32)

