from fastapi import APIRouter, UploadFile, File, HTTPException,Form
import os
from src.components.audioAnalysis.audioAnalysis import extract_audio_features
from pydub import AudioSegment
router = APIRouter()

@router.post("/analyzeAudio")
async def analyze_audio( assessment_id: str = Form(...),file: UploadFile = File(...)):
    try:
        original_path = "temp_audio.webm"
        wav_path = "temp_audio.wav"

        # Save the incoming webm file
        with open(original_path, "wb") as f:
            f.write(await file.read())

        # Convert webm -> wav using pydub
        audio = AudioSegment.from_file(original_path, format="webm")
        audio = audio.set_channels(1).set_frame_rate(16000)  # optional
        audio.export(wav_path, format="wav")

        # Now extract features
        features = extract_audio_features(assessment_id,wav_path)

        # Clean up files
        os.remove(original_path)
        os.remove(wav_path)

        return {"status": "success", "features": features}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))