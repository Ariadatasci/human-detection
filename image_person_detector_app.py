import streamlit as st
from PIL import Image
import numpy as np
import cv2

st.set_page_config(page_title="AI Person Detector (Lite)", layout="centered")

st.title("🧠 AI Image Person Detector (Lite)")
st.write("Upload an image. If a human face is detected, we'll classify it as **Person**.")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

@st.cache_resource
def load_face_detector():
    # OpenCV ships with Haar cascades; we’ll load the face cascade from OpenCV data.
    # Streamlit Cloud may not have the data path, so we bundle a fallback by downloading the xml if needed.
    # But most wheels include it; try default first:
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return cascade

face_cascade = load_face_detector()

def detect_faces(pil_image):
    # Convert PIL -> OpenCV BGR
    img = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )
    # Draw boxes for preview
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 180, 255), 3)
    return img, len(faces)

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded image", use_column_width=True)

    # Run detection
    boxed_img, count = detect_faces(image)
    st.subheader("🔍 Result")

    if count > 0:
        st.success(f"✅ Person detected (faces found: {count}).")
    else:
        st.warning("🚫 No person detected (no faces found).")

    # Show the annotated image
    st.image(boxed_img, caption="Detection preview", use_column_width=True)

st.caption("Tip: This lite version uses face detection. Try images with a clear, front-facing person for best results.")
