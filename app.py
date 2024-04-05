import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('model_save/recyclable_classifier_model.h5')  


classes = ['Non-Recyclable', 'Recyclable']

# Function to preprocess the image
def preprocess_image(image):
    # Resize the image to match the input shape of the model
    image = image.resize((200, 200))
    # Convert image to numpy array
    img_array = np.asarray(image)
    # Normalize the pixel values
    img_array = img_array / 255.0
    # Expand dimensions to create a batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Create Streamlit UI
def main():
    st.title('Recyclable Image Detection')

    # Upload image
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make prediction
        prediction = model.predict(processed_image)
        if prediction[0][0] > 0.5:
                print("Predicted: Recyclable")
                st.write('Prediction: Recyclable' )
        else:
                print("Predicted: Non-recyclable")
                st.write('Prediction: Non-Recyclable' )

        


if __name__ == '__main__':
    main()
