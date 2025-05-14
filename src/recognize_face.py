import cv2
import numpy as np
import os
from tkinter import filedialog
from tkinter import Tk
import time
from datetime import datetime

# Load trained model
model_path = os.path.join(os.path.dirname(__file__), "../models/trained_model.yml")
model = cv2.face.LBPHFaceRecognizer_create()
model.read(model_path)

# Load face detector
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Map label IDs to names
names = []  # List of names of people in dataset
dataset_path = os.path.join(os.path.dirname(__file__), "../dataset")
for person in os.listdir(dataset_path):
    if os.path.isdir(os.path.join(dataset_path, person)):
        names.append(person)

# Function to upload image
def upload_image():
    root = Tk()
    root.withdraw()  # Hide the Tkinter root window
    image_path = filedialog.askopenfilename()  # Open file dialog to select image
    return image_path

# Function to recognize face in the image
def recognize_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        print("[INFO] No faces detected!")
        return

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (200, 200))

        label, confidence = model.predict(face)

        if confidence < 100:  # Lower confidence threshold
            name = names[label]
            display_text = f"{name} ({round(confidence, 2)})"
            log_recognition(name)
        else:
            display_text = "Unknown"
            log_recognition("Unknown")

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Show the image with face recognition
    cv2.imshow("Face Recognition", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to log the timestamp when someone is recognized
def log_recognition(name):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open("recognition_log.txt", "a") as log_file:
        log_file.write(f"{timestamp} - Recognized: {name}\n")
    print(f"[INFO] Recognized: {name} at {timestamp}")

# Main function
if __name__ == "__main__":
    print("[INFO] Uploading an image...")
    image_path = upload_image()  # Upload an image
    if image_path:
        recognize_image(image_path)  # Recognize faces in the uploaded image
