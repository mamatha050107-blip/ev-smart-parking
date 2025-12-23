import gradio as gr
from ultralytics import YOLO
import pandas as pd

# Load trained model
model = YOLO("best.pt")

CLASS_NAMES = {
    0: "Empty Slot",
    1: "Occupied Slot"
}

def detect_and_summarize(image):
    results = model(image)

    boxes = results[0].boxes
    annotated_image = results[0].plot()

    data = []
    empty_count = 0
    occupied_count = 0

    if boxes is not None:
        for i, cls in enumerate(boxes.cls):
            label = CLASS_NAMES[int(cls)]
            data.append({
                "Slot ID": f"S{i+1}",
                "Status": label
            })
            if label == "Empty Slot":
                empty_count += 1
            else:
                occupied_count += 1

    summary_table = pd.DataFrame(data)

    stats = pd.DataFrame([{
        "Total Slots": empty_count + occupied_count,
        "Empty Slots": empty_count,
        "Occupied Slots": occupied_count
    }])

    return annotated_image, summary_table, stats

demo = gr.Interface(
    fn=detect_and_summarize,
    inputs=gr.Image(type="numpy", label="Upload Parking Image"),
    outputs=[
        gr.Image(label="Detected Parking Slots"),
        gr.Dataframe(label="Slot-wise Detection Table"),
        gr.Dataframe(label="Parking Summary")
    ],
    title="EV Smart Parking Slot Detection System",
    description=(
        "Upload a parking lot image. "
        "The system detects empty and occupied slots using a YOLOv8 model "
        "and provides both visual and tabular insights."
    ),
    allow_flagging="never"
)

demo.launch()
