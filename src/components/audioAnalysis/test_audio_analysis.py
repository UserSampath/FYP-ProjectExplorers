import os
from src.components.audioAnalysis.audioAnalysis import extract_audio_features

# Set path to your local audio file
audio_file_path = os.path.join("notebook", "data", "audioAnalysis", "harvard.wav")

# Call the feature extraction function
try:
    features = extract_audio_features(audio_file_path)
    print("Extracted Audio Features:")
    for key, value in features.items():
        print(f"{key}: {value}")
except Exception as e:
    print("Error while extracting features:", str(e))
