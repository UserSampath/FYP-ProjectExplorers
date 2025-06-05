from fastapi import APIRouter, UploadFile, File, HTTPException
import os
from src.components.audioAnalysis.audioAnalysis import extract_audio_features

router = APIRouter()

@router.post("/analyzeAudio")
async def analyze_audio(file: UploadFile = File(...)):
    try:
        file_path = "temp_audio.wav"
        with open(file_path, "wb") as f:
            f.write(await file.read())

        features = extract_audio_features(file_path)
        os.remove(file_path)

        return {"status": "success", "features": features}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
