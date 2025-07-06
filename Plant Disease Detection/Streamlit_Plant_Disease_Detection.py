import streamlit as st

#Page settings
st.set_page_config(page_title="Plant Disease Detector", layout="centered")
st.title("🌿 Plant Disease Detection using CNN")
st.markdown("Upload a plant leaf image and the AI model will predict the disease.")

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import os

#Loading the model
@st.cache_resource
def load_cnn_model():
    return load_model("plant_disease_model_keras.h5")

model = load_cnn_model()

#Defining class labels
class_labels = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Corn___Cercospora_leaf_spot",
    "Corn___Common_rust",
    "Corn___healthy",
    "Tomato___Late_blight",
    "Tomato___Septoria_leaf_spot",
    "Tomato___healthy"
]

#Upload section
uploaded_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    #Preprocessinh the image
    img = img.resize((128, 128))
    img_array = keras_image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"✅ Predicted Disease: **{predicted_class}**")
    st.info(f"📊 Confidence: {confidence:.2f}%")
