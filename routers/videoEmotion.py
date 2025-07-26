from fastapi import APIRouter, UploadFile, File, HTTPException
from tempfile import NamedTemporaryFile
from src.pipeline.videoEmotion.videoEmotion import analyze_video
from src.pipeline.audioEmotion.audioEmotion import predict_emotion

from src.schemas.schemas import APIResponse
import subprocess
import os
router = APIRouter()

@router.post("/analyzeVideo/{assessment_id}", response_model=APIResponse)
async def analyze_video_endpoint(assessment_id: str, file: UploadFile = File(...)):
    try:
        # Save uploaded video file
        with NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(await file.read())
            temp_video_path = temp_video.name

        # Extract audio from video
        audio_output_path = "data/tempAudio.wav"
        os.makedirs("data", exist_ok=True)

        ffmpeg_command = [
            "ffmpeg", "-i", temp_video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            audio_output_path, "-y"
        ]
        subprocess.run(ffmpeg_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Predict audio emotion
        try:
            emotion = predict_emotion(audio_output_path)
            print(f"🎧 Predicted Audio Emotion: {emotion}")
        except Exception as audio_err:
            emotion = "no emotion detected in audio"
            print(f"⚠️ Error during audio emotion prediction: {audio_err}")

        # Analyze video WITH audio emotion passed in
        result = analyze_video(temp_video_path, assessment_id, emotion)

        # Clean up
        os.remove(temp_video_path)
        if os.path.exists(audio_output_path):
            os.remove(audio_output_path)

        return {
            "data": result,
            "status": "200",
            "success": True,
            "message": "Analyze video successfully"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

