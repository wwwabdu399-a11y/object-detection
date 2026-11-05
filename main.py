import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io
import os

st.set_page_config(
    page_title="Traffic Sign Detection",
    page_icon="🚦",
    layout="centered"
)

st.title("Traffic Sign Detection")
st.write("Upload an image to detect traffic signs using YOLOv8.")

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "traffic_sign_model.pt"
)


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


if not os.path.exists(MODEL_PATH):
    st.error("Model file not found: traffic_sign_model.pt")
    st.stop()

model = load_model()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

confidence = st.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    if st.button("Detect Traffic Signs"):

        with st.spinner("Detecting traffic signs..."):

            results = model.predict(
                source=image,
                conf=confidence,
                imgsz=640,
                device="cpu",
                verbose=False
            )

            result = results[0]

            detected_image = result.plot()

            output_image = Image.fromarray(
                detected_image[:, :, ::-1]
            )

            st.subheader("Detection Results")

            st.image(
                output_image,
                use_container_width=True
            )

            detection_count = len(result.boxes)

            st.success(
                f"Detection completed! "
                f"Found {detection_count} objects."
            )

            st.metric(
                "Detected Objects",
                detection_count
            )

            image_buffer = io.BytesIO()

            output_image.save(
                image_buffer,
                format="PNG"
            )

            st.download_button(
                label="Download Detection Image",
                data=image_buffer.getvalue(),
                file_name="traffic_sign_detection.png",
                mime="image/png"
            )