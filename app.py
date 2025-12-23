import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import tempfile

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="EV Smart Parking – Slot Occupancy Detection",
    layout="wide"
)

st.title("EV Smart Parking – Slot Occupancy Detection")
st.write("Image and Video based parking slot occupancy detection using YOLOv8")

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# -----------------------------
# MODE SELECTION
# -----------------------------
mode = st.radio("Select Input Type", ["Image", "Video"])

# =====================================================
# IMAGE MODE
# =====================================================
if mode == "Image":
    uploaded_file = st.file_uploader("Upload parking image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)

        results = model.predict(img_array, conf=0.25)
        result = results[0]

        annotated = result.plot()
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        class_names = model.names
        counts = {"space-empty": 0, "space-occupied": 0}

        for cls in result.boxes.cls:
            label = class_names[int(cls)]
            if label in counts:
                counts[label] += 1

        st.image(annotated, caption="Detection Output", use_column_width=True)

        df = pd.DataFrame({
            "Status": ["Empty", "Occupied"],
            "Count": [counts["space-empty"], counts["space-occupied"]]
        })
        st.table(df)

# =====================================================
# VIDEO MODE
# =====================================================
if mode == "Video":
    uploaded_video = st.file_uploader("Upload parking CCTV video", type=["mp4", "avi", "mov"])

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        empty_count = 0
        occupied_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, conf=0.25, verbose=False)
            result = results[0]

            annotated = result.plot()

            class_names = model.names
            empty_count = 0
            occupied_count = 0

            for cls in result.boxes.cls:
                label = class_names[int(cls)]
                if label == "space-empty":
                    empty_count += 1
                elif label == "space-occupied":
                    occupied_count += 1

            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            stframe.image(annotated, use_column_width=True)

        cap.release()

        st.subheader("Final Slot Count")
        df = pd.DataFrame({
            "Status": ["Empty", "Occupied"],
            "Count": [empty_count, occupied_count]
        })
        st.table(df)
