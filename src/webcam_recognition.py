import cv2
import os
import numpy as np
import speech_recognition as sr
import pyttsx3
import threading
from datetime import datetime

# Initialize TTS engine
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Load face model
model = cv2.face.LBPHFaceRecognizer_create()
model.read("models/trained_model.yml")

# Load face detector
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load dataset names
dataset_path = "dataset"
names = [name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))]

# Voice control flags
exit_program = False
start_recognition = False

def log_recognition(name):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open("recognition_log.txt", "a") as f:
        f.write(f"{timestamp} - Recognized: {name}\n")
    print(f"[INFO] Recognized {name} at {timestamp}")

def listen_for_commands():
    global exit_program, start_recognition
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print("[VOICE] Say 'start', 'stop', or 'exit'...")
        speak("Voice control activated. Say start, stop, or exit.")
        while not exit_program:
            try:
                audio = recognizer.listen(source, timeout=5)
                command = recognizer.recognize_google(audio).lower()
                print(f"[VOICE] You said: {command}")
                if "start" in command:
                    start_recognition = True
                    speak("Starting recognition")
                elif "stop" in command:
                    start_recognition = False
                    speak("Stopping recognition")
                elif "exit" in command:
                    exit_program = True
                    speak("Exiting system")
                    break
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                print(f"[ERROR] Voice command failed: {e}")

def recognize_from_webcam_with_voice():
    global exit_program, start_recognition
    exit_program = False
    start_recognition = False

    cap = cv2.VideoCapture(0)
    listener_thread = threading.Thread(target=listen_for_commands)
    listener_thread.start()

    while not exit_program:
        ret, frame = cap.read()
        if not ret:
            continue

        if start_recognition:
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
                    speak(name)
                else:
                    text = "Unknown"
                    log_recognition("Unknown")
                    speak("Unknown")
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("Webcam Recognition (Voice Controlled)", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    exit_program = True
    listener_thread.join()
