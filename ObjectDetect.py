import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 1️⃣ Page Configuration
st.set_page_config(
    page_title="OBJECT Detection AI",
    page_icon="⛑️",
    layout="centered"
)

# 2️⃣ Load YOLO Model (cached for performance)
@st.cache_resource
def load_yolo_model():
    return YOLO("Object.pt")  # model must be in repo root

model = load_yolo_model()

# 3️⃣ UI Layout
st.title("OBJECT Detection System")
st.write("Upload an image to detect if a OBJECT is being worn.")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.info("Original Image")
        st.image(image, use_column_width=True)

    if st.button("Run OBJECT Detection"):

        with st.spinner("Analyzing image..."):

            # Run prediction
            results = model.predict(image)

            # Get plotted image directly (no OpenCV needed)
            res_plotted = results[0].plot()

            with col2:
                st.success("Detection Result")
                st.image(res_plotted, caption="Processed Image", use_column_width=True)

            # Detection logic
            detections = results[0].boxes

            if detections is not None and len(detections) > 0:

                detected_names = [
                    model.names[int(cls)] for cls in detections.cls
                ]

                st.write("Detected Classes:", detected_names)

                detected_lower = [
                    name.lower().replace("_", " ").strip()
                    for name in detected_names
                ]

                if "no plate" in detected_lower:
                    st.success("### ✅ Prediction: OBJECT Detected")
                    st.balloons()
                else:
                    st.error("### ❌ Prediction: No OBJECT Detected")
                    st.warning("Only other objects were found in the frame.")

            else:
                st.error("### ❌ Prediction: No OBJECT Detected")
                st.info("The model could not identify any objects in this image.")

            st.metric(
                label="Total Objects Found",
                value=len(detections) if detections is not None else 0
            )
