import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import cv2
import os
import numpy as np
import speech_recognition as sr
import pyttsx3

# Initialize speech engine
engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Load model
model = cv2.face.LBPHFaceRecognizer_create()
model.read("models/trained_model.yml")

# Load face classifier
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load names
dataset_path = "dataset"
names = [name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))]

# Global flag
webcam_running = False

# Logging
from datetime import datetime
def log_recognition(name):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open("recognition_log.txt", "a") as f:
        f.write(f"{timestamp} - Recognized: {name}\n")
    print(f"[INFO] Recognized {name} at {timestamp}")

# Face recognition from image
def recognize_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        messagebox.showinfo("Result", "No faces detected!")
        return

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (200, 200))
        label, confidence = model.predict(face)

        if confidence < 100:
            name = names[label]
            display_text = f"{name} ({round(confidence, 2)})"
            log_recognition(name)
            speak(f"Hello, {name}")
        else:
            display_text = "Unknown"
            log_recognition("Unknown")
            speak("Unknown person detected")

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow("Recognition", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Webcam recognition
def recognize_from_webcam():
    global webcam_running
    webcam_running = True
    cap = cv2.VideoCapture(0)

    while webcam_running:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))
            label, confidence = model.predict(face)
            if confidence < 100:
                name = names[label]
                text = f"{name} ({round(confidence, 2)})"
                log_recognition(name)
                speak(f"Hello, {name}")
            else:
                text = "Unknown"
                log_recognition("Unknown")
                speak("Unknown person detected")

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("Webcam Recognition", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# Voice commands
def listen_voice_commands():
    global webcam_running
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    speak("Voice control activated. Say start, stop, or exit.")

    while True:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            print("Listening...")
            try:
                audio = recognizer.listen(source, timeout=5)
                command = recognizer.recognize_google(audio).lower()
                print(f"[VOICE] Command: {command}")

                if "start" in command and not webcam_running:
                    threading.Thread(target=recognize_from_webcam, daemon=True).start()
                elif "stop" in command:
                    webcam_running = False
                    speak("Webcam recognition stopped.")
                elif "exit" in command:
                    webcam_running = False
                    speak("Exiting application.")
                    window.quit()
                    break
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                speak("Sorry, I didn't catch that.")
            except Exception as e:
                print("[ERROR]", e)
                speak("There was an error processing your voice.")

# GUI functions
def upload_and_recognize():
    image_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.jpg;*.png")])
    if image_path:
        try:
            recognize_image(image_path)
        except Exception as e:
            messagebox.showerror("Error", f"Image recognition failed: {e}")
    else:
        messagebox.showwarning("No File", "No image file selected!")

# Create GUI
window = tk.Tk()
window.title("Smart Face Recognition")
window.geometry("350x250")

tk.Label(window, text="Choose Input Method:", font=("Helvetica", 12)).pack(pady=10)

btn_webcam = tk.Button(window, text="Start Webcam", command=lambda: threading.Thread(target=recognize_from_webcam, daemon=True).start(), width=20, height=2)
btn_webcam.pack(pady=5)

btn_upload = tk.Button(window, text="Upload Image", command=upload_and_recognize, width=20, height=2)
btn_upload.pack(pady=5)

btn_voice = tk.Button(window, text="Voice Control", command=lambda: threading.Thread(target=listen_voice_commands, daemon=True).start(), width=20, height=2)
btn_voice.pack(pady=5)

window.mainloop()
