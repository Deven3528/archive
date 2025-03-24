import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

def load_model():
    model = tf.keras.models.load_model("drmodel.h5")  # Load your trained model
    return model

def preprocess_image(image):
    image = image.convert("RGB")  # Ensure it has 3 channels (RGB)
    image = image.resize((224, 224))  # Resize to model input shape
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension (1, 224, 224, 3)
    return image


def predict_severity(model, image):
    prediction = model.predict(image)[0]
    severity_levels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
    class_index = np.argmax(prediction)
    confidence = round(prediction[class_index] * 100, 2)
    return severity_levels[class_index], confidence

# Load the model
model = load_model()

# Streamlit UI Design
st.set_page_config(page_title="Diabetic Retinopathy Detector", layout="centered")

# Custom Styling
st.markdown(
    """
    <style>
        .main { background-color: #f4f4f4; }
        h1 { color: #4CAF50; text-align: center; }
        .stButton>button { background-color: #4CAF50; color: white; font-size: 18px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and UI Elements
st.image("https://tse2.mm.bing.net/th?id=OIP.mjlzlO2y3nkybFRy1cy-8AHaEK&pid=Api&P=0&h=180", use_column_width=True)
st.title("🩺 Diabetic Retinopathy Prediction")
st.write("Upload a retinal scan image to detect Diabetic Retinopathy severity.")

# File Uploader
uploaded_file = st.file_uploader("Upload Retinal Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Processing... 🔄")
    
    # Preprocess and Predict
    processed_image = preprocess_image(image)
    severity, confidence = predict_severity(model, processed_image)
    
    # Show Prediction
    st.subheader(f"🧐 Severity: {severity}")
    st.progress(confidence / 100)
    st.success(f"🔥 Confidence Score: {confidence}%")
