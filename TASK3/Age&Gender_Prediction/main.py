import streamlit as st
import cv2
import numpy as np
import os

# Model files - make sure these are in your project folder
FACE_PROTO = "models/opencv_face_detector.pbtxt"
FACE_MODEL = "models/opencv_face_detector_uint8.pb"
AGE_PROTO = "models/age_deploy.prototxt"
AGE_MODEL = "models/age_net.caffemodel"
GENDER_PROTO = "models/gender_deploy.prototxt"
GENDER_MODEL = "models/gender_net.caffemodel"

AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']
MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Load models
face_net = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)

def detect_faces(image):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), MEAN_VALUES, swapRB=True)
    face_net.setInput(blob)
    detections = face_net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            boxes.append(box.astype(int))
    return boxes

def predict_age_gender(image):
    boxes = detect_faces(image)
    for box in boxes:
        x1, y1, x2, y2 = box
        face = image[max(0,y1):min(y2,image.shape[0]-1), max(0,x1):min(x2,image.shape[1]-1)]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MEAN_VALUES, swapRB=False)
        
        gender_net.setInput(blob)
        gender = GENDER_LIST[gender_net.forward().argmax()]
        age_net.setInput(blob)
        age = AGE_LIST[age_net.forward().argmax()]
        
        label = f"{gender}, {age}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    return image

def list_images(directory):
    return [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Streamlit UI
st.title("🧠 Age & Gender Prediction - Select Image from Project Folder")

IMAGE_DIR = "./images"  # Folder path for your images

if not os.path.exists(IMAGE_DIR):
    st.error(f"Image directory '{IMAGE_DIR}' not found! Please create it and add images.")
else:
    image_files = list_images(IMAGE_DIR)
    if not image_files:
        st.warning(f"No images found in '{IMAGE_DIR}'. Please add some .jpg/.png files.")
    else:
        selected_img = st.selectbox("Select an image", image_files)
        if selected_img:
            img_path = os.path.join(IMAGE_DIR, selected_img)
            img = cv2.imread(img_path)
            if img is not None:
                output = predict_age_gender(img.copy())
                st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), caption=f"Predictions on {selected_img}")
            else:
                st.error("Failed to load image.")
