import sys
import os
import warnings

# Add pyAudioAnalysis folder to Python path
current_dir = os.path.dirname(__file__)
pyaudio_path = os.path.join(current_dir, "pyAudioAnalysis")
sys.path.append(pyaudio_path)

# Imports
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
import librosa
import numpy as np
from src.controllers.suggestionController import createSuggestions

def extract_audio_features(assessment_id,file_path):
    warnings.filterwarnings("ignore")

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

    # --- Feature: Silence Ratio (improved using RMS) ---
    frame_length = 1024
    hop_length = 512
    rms_energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    threshold = np.percentile(rms_energy, 10)
    silent_frames = rms_energy < threshold
    silence_ratio = np.sum(silent_frames) / len(rms_energy)

    # --- Feature: Speaking Rate (Tempo Proxy) using onsets ---
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames')
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    duration_minutes = len(y) / sr / 60.0
    speaking_rate = len(onset_times) / duration_minutes if duration_minutes > 0 else 0

    # --- Additional Features using pyAudioAnalysis ---
    [fs, x] = audioBasicIO.read_audio_file(file_path)
    x = audioBasicIO.stereo_to_mono(x)
    F, f_names = ShortTermFeatures.feature_extraction(x, fs, 0.050 * fs, 0.025 * fs)

    zcr = np.mean(F[0])  # Zero Crossing Rate
    spectral_centroid = np.mean(F[4])  # Spectral Centroid

    # --- Generate Suggestions ---
    suggestions = []

    if avg_loudness < 0.02:
        suggestions.append("Try speaking louder; your voice was too soft.")
    elif avg_loudness > 0.1:
        suggestions.append("Your voice was a bit loud; try speaking more calmly.")

    if energy_std < 0.01:
        suggestions.append("Your speech energy is too flat; consider adding more emphasis.")

    if mean_pitch < 1000:
        suggestions.append("Your pitch is quite low; consider varying it to sound more engaging.")
    elif mean_pitch > 2000:
        suggestions.append("Your pitch is high; try to moderate it for clarity.")

    if pitch_std < 20:
        suggestions.append("Your pitch variation is limited; try to use more expressive intonation.")

    if speaking_rate < 180:
        suggestions.append("Your speaking rate was slow; consider speeding up slightly.")
    elif speaking_rate > 250:
        suggestions.append("You are speaking too fast; slow down to be more understandable.")

    if silence_ratio > 0.3:
        suggestions.append("There are many silent pauses; practice to improve fluency.")
    elif silence_ratio < 0.05:
        suggestions.append("You barely paused; add natural breaks for better pacing.")

    if zcr > 0.4:
        suggestions.append("There might be background noise; ensure a quiet recording environment.")

    if len(suggestions) > 0:
        createSuggestions(assessment_id, suggestions)
    # Final return object
    return {
        "average_loudness": float(avg_loudness),
        "energy_std": float(energy_std),
        "mean_pitch": float(mean_pitch),
        "pitch_std": float(pitch_std),
        "speaking_rate": float(speaking_rate),
        "silence_ratio": float(silence_ratio),
        "zcr": float(zcr),
        "spectral_centroid": float(spectral_centroid),
        "suggestions": suggestions
    }
