from fastapi import APIRouter, UploadFile, File, HTTPException
from tempfile import NamedTemporaryFile
from src.pipeline.videoEmotion.videoEmotion import analyze_video
from src.schemas.schemas import APIResponse

import os
router = APIRouter()

@router.post("/analyzeVideo/{assessment_id}",response_model=APIResponse)
async def analyze_video_endpoint(assessment_id:str,file: UploadFile = File(...)):
    try:
        # Save video temporarily
        with NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(await file.read())
            temp_video_path = temp_video.name

        result = analyze_video(temp_video_path,assessment_id)

        os.remove(temp_video_path)
        return { "data":result,"status": "200","success":True,"message":"Analyze video successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
