import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import urllib.request
import os

# ------------------------
# 1. Page config
# ------------------------
st.set_page_config(page_title="Object Detection AI", page_icon="⛑️", layout="centered")

# ------------------------
# 2. Download model
# ------------------------
MODEL_URL = "https://huggingface.co/ShumasChohan/Object-Detection-Using-AI/resolve/main/Object.pt"
MODEL_PATH = "Object.pt"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        st.success("Model downloaded!")

# ------------------------
# 3. Load YOLO model
# ------------------------
@st.cache_resource
def load_yolo_model(path):
    return YOLO(path)

model = load_yolo_model(MODEL_PATH)

# ------------------------
# 4. UI Layout
# ------------------------
st.title("Object Detection System")
st.write("Upload an image to detect objects.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original Image", use_column_width=True)

    if st.button("Run Detection"):
        with st.spinner("Analyzing image..."):
            results = model.predict(image)

            # Get plotted result
            res_plotted = results[0].plot()
            
            # Convert to RGB (from BGR)
            import cv2
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

            with col2:
                st.image(res_rgb, caption="Detection Result", use_column_width=True)

            detections = results[0].boxes
            if len(detections) > 0:
                detected_names = [model.names[int(cls)] for cls in detections.cls]
                st.write("Detected Classes:", detected_names)

                if any("no plate" in name.lower() for name in detected_names):
                    st.success("✅ Prediction: OBJECT Detected")
                    st.balloons()
                else:
                    st.error("❌ Prediction: No OBJECT Detected")
            else:
                st.error("❌ Prediction: No OBJECT Detected")

            st.metric(label="Total Objects Found", value=len(detections))
