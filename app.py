import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

st.set_page_config(page_title="EV Smart Parking", layout="centered")

st.title("🚗 EV Smart Parking – Slot Occupancy Detection")

st.write("""
This system detects parking slot occupancy using a YOLOv8 deep learning model.
Upload a parking lot image to analyze slot availability.
""")

# Load model (cached)
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded_file = st.file_uploader("Upload Parking Lot Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        results = model(tmp.name)

    st.subheader("Detection Result")
    result_img = results[0].plot()
    st.image(result_img, use_column_width=True)

    # Count slots
    empty = 0
    occupied = 0

    for box in results[0].boxes:
        cls = int(box.cls[0])
        if cls == 0:
            empty += 1
        else:
            occupied += 1

    st.subheader("Slot Summary")
    st.write(f"🟩 Empty Slots: {empty}")
    st.write(f"🟥 Occupied Slots: {occupied}")
