# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 16:45:18 2024

@author: shan2
"""

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# Load the YOLOv8 model
model = YOLO(r"C:/Users/adity/Downloads/project/runs/detect/train3/weights/best.pt")  # Replace with your trained model path if needed

# Streamlit App
st.title("YOLOv8 Object Detection App")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Running inference...")

    # Convert to OpenCV format
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Perform inference
    results = model.predict(image_bgr)

    # Display results
    results_img = results[0].plot()  # Use plot method to visualize results
    st.image(results_img, caption='Detected Objects', use_column_width=True)
