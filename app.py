import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="EV Smart Parking – Slot Occupancy Detection",
    layout="wide"
)

st.title("🚗 EV Smart Parking – Slot Occupancy Detection")
st.write(
    """
    This application detects **parking slot occupancy** using a trained YOLOv8 model.
    It is designed for **industry-scale smart parking systems**.
    """
)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a parking lot image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.subheader("📥 Input Image")
    st.image(image, use_column_width=True)

    # -----------------------------
    # RUN INFERENCE
    # -----------------------------
    with st.spinner("Detecting parking slots..."):
        results = model.predict(
            source=img_array,
            conf=0.25,
            save=False
        )

    result = results[0]

    # -----------------------------
    # PROCESS RESULTS
    # -----------------------------
    boxes = result.boxes
    class_names = model.names

    counts = {"space-empty": 0, "space-occupied": 0}

    for cls in boxes.cls:
        label = class_names[int(cls)]
        if label in counts:
            counts[label] += 1

    total_slots = counts["space-empty"] + counts["space-occupied"]

    # -----------------------------
    # DRAW OUTPUT IMAGE
    # -----------------------------
    annotated_img = result.plot()
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    st.subheader("📊 Detection Output")
    st.image(annotated_img, use_column_width=True)

    # -----------------------------
    # SUMMARY METRICS
    # -----------------------------
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Slots", total_slots)
    col2.metric("Empty Slots", counts["space-empty"])
    col3.metric("Occupied Slots", counts["space-occupied"])

    # -----------------------------
    # TABLE VIEW (INDUSTRY LIKES THIS)
    # -----------------------------
    st.subheader("📋 Slot Occupancy Summary")

    df = pd.DataFrame({
        "Slot Status": ["Empty", "Occupied"],
        "Count": [counts["space-empty"], counts["space-occupied"]]
    })

    st.table(df)

    # -----------------------------
    # FOOTER NOTE
    # -----------------------------
    st.info(
        "This model was trained offline and deployed for real-time inference. "
        "Future extensions include EV charging port health monitoring and video analytics."
    )

else:
    st.warning("Please upload a parking lot image to start detection.")
