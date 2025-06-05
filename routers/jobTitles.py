from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.utils import fetch_and_save_job_titles

router = APIRouter()

class SourceTypeRequest(BaseModel):
    source_type: str

@router.post("/fetchJobTitles")
def fetch_job_titles(request: SourceTypeRequest):
    try:
        fetch_and_save_job_titles(request.source_type)
        return {"status": "success", "message": "Job titles fetched and saved successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
