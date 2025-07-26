import os
import numpy as np
import librosa
import joblib

# Load model and label encoder once
model = joblib.load("artifact/audioEmotion/xgb_emotion_model.pkl")
le = joblib.load("artifact/audioEmotion/label_encoder.pkl")

# Feature extraction
def extract_features(file_path, mfcc=True, chroma=True, mel=True):
    y, sr = librosa.load(file_path, sr=None)
    result = np.array([])

    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))

    if chroma:
        stft = np.abs(librosa.stft(y))
        chroma_feat = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
        result = np.hstack((result, chroma_feat))

    if mel:
        mel_feat = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
        result = np.hstack((result, mel_feat))

    return result

# Predict emotion from a file
def predict_emotion(audio_path):
    if os.path.exists(audio_path):
        try:
            feature = extract_features(audio_path).reshape(1, -1)
            prediction = model.predict(feature)
            emotion = le.inverse_transform(prediction)[0]
            return emotion
        except Exception as e:
            raise RuntimeError(f"Error while processing audio: {e}")
    else:
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
