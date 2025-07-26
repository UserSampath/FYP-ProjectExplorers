import os
import librosa
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Set your dataset path
dataset_path = "notebook/data/audioEmotion/ravdess"

# Emotion mapping
emotions = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

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

# Load data
def load_data(base_path):
    X, y = [], []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".wav"):
                emotion_code = file.split("-")[2]
                emotion = emotions.get(emotion_code)
                if emotion:
                    try:
                        feature = extract_features(os.path.join(root, file))
                        X.append(feature)
                        y.append(emotion)
                    except Exception as e:
                        print(f"Error processing {file}: {e}")
    return np.array(X), np.array(y)

# Load and preprocess
print("Loading and extracting features...")
X, y = load_data(dataset_path)
le = LabelEncoder()
y = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
print("Training model...")
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Save model and encoder
save_path = "artifact/audioEmotion"
os.makedirs(save_path, exist_ok=True)
model_file = os.path.join(save_path, "xgb_emotion_model.pkl")
encoder_file = os.path.join(save_path, "label_encoder.pkl")

joblib.dump(model, model_file)
joblib.dump(le, encoder_file)

print(f"\n✅ Model saved to: {model_file}")
print(f"✅ Label encoder saved to: {encoder_file}")
