from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.pipeline.languageProficiency.languageProficiency import predict_all_scores

router = APIRouter()

class TextRequest(BaseModel):
    text: str

@router.post("/predictLanguageScore")
def predict_language_score(request: TextRequest):
    try:
        scores = predict_all_scores(request.text)
        return {"status": "success", "predicted_score": scores}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
