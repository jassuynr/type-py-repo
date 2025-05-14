import os
import cv2
import numpy as np

# Path to dataset
dataset_path = os.path.join(os.path.dirname(__file__), "../dataset")
dataset_path = os.path.abspath(dataset_path)

print("[DEBUG] Looking for dataset folder at:", dataset_path)

# Load face detector
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Lists to hold training data
faces = []
labels = []
label_id = 0

# Loop through each person in dataset
for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)

    if not os.path.isdir(person_path):
        continue

    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        image = cv2.imread(image_path)

        if image is None:
            print(f"[WARN] Could not read {image_path}, skipping.")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces_rect = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces_rect:
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (200, 200))  # normalize size
            faces.append(face)
            labels.append(label_id)

    print(f"[INFO] Processed images for: {person}")
    label_id += 1

# Convert to NumPy arrays
faces = np.array(faces)
labels = np.array(labels)

# Train the recognizer
print("[INFO] Training model...")
model = cv2.face.LBPHFaceRecognizer_create()
model.train(faces, labels)

# Save the trained model
model_dir = os.path.join(os.path.dirname(__file__), "../models")
os.makedirs(model_dir, exist_ok=True)
model.save(os.path.join(model_dir, "trained_model.yml"))

print("[✔] Training complete and model saved!")
