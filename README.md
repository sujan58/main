# Waste Classification Project

This project aims to classify waste images into recyclable and non-recyclable categories using deep learning techniques. The dataset consists of images crawled from Google Images and is categorized into recyclable and non-recyclable folders.

## Overview

The project involves the following components:

**DataSet Link: `https://drive.google.com/drive/folders/1cKSsaI4DvTtIH9-6-3edIZ_IVRxQ84O7?usp=sharing`**
1. Data Collection: Images of recyclable and non-recyclable items were collected from Google Images.
2. Data Preprocessing: Images were processed, including resizing, cropping, and labeling.
3. Model Development: A convolutional neural network (CNN) model was developed using TensorFlow and Keras to classify the images.
4. Model Evaluation: The model's performance was evaluated using accuracy and loss metrics.
5. Model Deployment: A Streamlit web application was developed to allow users to interactively classify waste images.

## Dataset

- **Recyclable Images**: Contains images of recyclable items such as plastic bottles, aluminum cans, etc.
- **Non-Recyclable Images**: Contains images of non-recyclable items such as food waste, styrofoam, etc.

## Model Architecture

The CNN model architecture consists of several convolutional and pooling layers followed by fully connected layers. Dropout layers were included to prevent overfitting.

## Requirements

To run the project, you'll need the following dependencies:

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Pillow
- Streamlit

You can install these dependencies using the `requirements.txt` file provided in the repository.

## Running the Project

1. Clone the repository to your local machine.
2. Install the dependencies using the following command:
`pip install -r requirements.txt`

3. Run the Jupyter Notebook `model.ipynb` to train and evaluate the model.
4. To run the Streamlit app, execute the following command:

  `streamlit run app.py`


