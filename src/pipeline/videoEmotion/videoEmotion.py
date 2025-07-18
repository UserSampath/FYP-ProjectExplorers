import cv2
import numpy as np
from deepface import DeepFace
from collections import Counter
import mediapipe as mp
import os
from scipy.ndimage import gaussian_filter1d

# --- Preprocessing for Low Quality Frames ---
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE for local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Optional: slight Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    return cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)

# --- MediaPipe Setup (Set static_image_mode=False for video robustness) ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,  # Important for continuous frames
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# --- Landmark Extraction ---
def get_landmarks(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    return results.multi_face_landmarks[0] if results.multi_face_landmarks else None

# --- Emotion Detection with relaxed fallback and resizing ---
def analyze_emotion(frame):
    try:
        # Resize frame to ensure face is large enough for detectors
        resized = cv2.resize(frame, (480, 480))
        try:
            faces = DeepFace.extract_faces(resized, detector_backend='retinaface', enforce_detection=True)
            analysis_target = faces[0]["face"] if faces else resized
            result = DeepFace.analyze(analysis_target, actions=['emotion'], enforce_detection=True)
        except:
            result = DeepFace.analyze(resized, actions=['emotion'], enforce_detection=False)

        return result[0]['dominant_emotion'] if 'dominant_emotion' in result[0] else "No Face"
    except Exception as e:
        return "No Face"

# --- Stability & Confidence ---
def calculate_stability_score(emotions):
    changes = sum(1 for i in range(1, len(emotions)) if emotions[i] != emotions[i - 1])
    return 1 - (changes / max(len(emotions) - 1, 1))

emotion_confidence_map = {
    "happy": 90, "surprise": 80, "neutral": 70, "calm": 70,
    "sad": 40, "fear": 30, "angry": 35, "disgust": 30, "No Face": 0
}

# --- Main Analyzer ---
def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    detected_emotions = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process every 10th frame
        if frame_count % 10 == 0:
            enhanced = preprocess_frame(frame)
            emotion = analyze_emotion(enhanced)
            detected_emotions.append(emotion)

        frame_count += 1

    cap.release()

    # Smooth predictions using a moving average filter
    confidence_scores = [emotion_confidence_map.get(e, 50) for e in detected_emotions]
    smoothed_confidence = gaussian_filter1d(confidence_scores, sigma=1)

    # Average and stability
    emotion_avg_conf = float(np.mean(smoothed_confidence)) if smoothed_confidence.size > 0 else 0
    stability = calculate_stability_score(detected_emotions)
    final_conf = (0.8 * emotion_avg_conf) + (0.2 * stability * 100)

    # Penalty for negatives
    negatives = ['fear', 'disgust', 'angry', 'No Face']
    penalty = sum(1 for e in detected_emotions if e in negatives) / len(detected_emotions)
    adjusted_conf = final_conf * (1 - 0.3 * penalty)
    adjusted_conf = max(adjusted_conf, 0)

    # Label
    if adjusted_conf >= 80:
        label = "High Confidence"
    elif adjusted_conf >= 60:
        label = "Moderate Confidence"
    else:
        label = "Low Confidence"

    return {
        "emotions": detected_emotions,
        "average_confidence": round(emotion_avg_conf, 2),
        "stability_score": round(stability, 2),
        "final_confidence": round(final_conf, 2),
        "penalty_adjusted_confidence": round(adjusted_conf, 2),
        "confidence_label": label
    }
