import sys
import os

# Add pyAudioAnalysis folder to Python path
current_dir = os.path.dirname(__file__)
pyaudio_path = os.path.join(current_dir, "pyAudioAnalysis")
sys.path.append(pyaudio_path)

# Now you can import
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures

import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

def extract_audio_features(file_path):
    # --- Load audio ---
    y, sr = librosa.load(file_path, sr=16000)

    # --- Feature: Loudness (RMS Energy) ---
    rms = librosa.feature.rms(y=y)[0]
    avg_loudness = np.mean(rms)
    energy_std = np.std(rms)

    # --- Feature: Pitch (Fundamental Frequency) ---
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitches = pitches[magnitudes > np.median(magnitudes)]
    mean_pitch = np.mean(pitches) if len(pitches) > 0 else 0
    pitch_std = np.std(pitches) if len(pitches) > 0 else 0

    # --- Feature: Speaking Rate (Tempo Proxy) ---
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]

    # --- Feature: Silence Ratio ---
    frame_length = 1024
    hop_length = 512
    energy = np.array([
        np.sum(np.abs(y[i:i+frame_length]**2))
        for i in range(0, len(y), hop_length)
    ])
    threshold = np.percentile(energy, 10)
    silent_frames = energy < threshold
    silence_ratio = np.sum(silent_frames) / len(energy)

    # --- Additional Features using pyAudioAnalysis ---
    [fs, x] = audioBasicIO.read_audio_file(file_path)
    x = audioBasicIO.stereo_to_mono(x)
    F, f_names = ShortTermFeatures.feature_extraction(x, fs, 0.050 * fs, 0.025 * fs)

    zcr = np.mean(F[0])  # Zero Crossing Rate
    spectral_centroid = np.mean(F[4])  # Spectral Centroid

    return {
        "average_loudness": float(avg_loudness),
        "energy_std": float(energy_std),
        "mean_pitch": float(mean_pitch),
        "pitch_std": float(pitch_std),
        "speaking_rate": float(tempo),
        "silence_ratio": float(silence_ratio),
        "zcr": float(zcr),
        "spectral_centroid": float(spectral_centroid)
    }
