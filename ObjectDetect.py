import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="OBJECT Detection AI", page_icon="⛑️", layout="centered")

@st.cache_resource
def load_yolo_model():
    # Use relative path to model in repo
    model_path = "Object.pt"
    return YOLO(model_path)

# Load model
model = load_yolo_model()

# 2. UI Layout
st.title("OBJECT Detection System")
st.write("Upload an image to detect if a OBJECT is being worn.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.info("Original Image")
        st.image(image, use_column_width=True)

    if st.button('Run OBJECT Detection'):
        with st.spinner('Analyzing image...'):

            results = model.predict(image)

            res_plotted = results[0].plot()
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

            with col2:
                st.success("Detection Result")
                st.image(res_rgb, caption='Processed Image', use_column_width=True)

            detections = results[0].boxes

            if len(detections) > 0:
                detected_names = [model.names[int(cls)] for cls in detections.cls]
                st.write("Detected Classes:", detected_names)

                detected_lower = [name.lower().replace("_", " ").strip() for name in detected_names]

                if "no plate" in detected_lower:
                    st.success("### ✅ Prediction: OBJECT Detected")
                    st.balloons()
                else:
                    st.error("### ❌ Prediction: No OBJECT Detected")
                    st.warning("Only other objects were found in the frame.")
            else:
                st.error("### ❌ Prediction: No OBJECT Detected")
                st.info("The model could not identify any objects in this image.")

            st.metric(label="Total Objects Found", value=len(detections))
