from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.pipeline.languageProficiency.languageProficiency import predict_all_scores
from src.controllers.languageProficiencyController import update_assessment
router = APIRouter()

class TextRequest(BaseModel):
    text: str
    assessment_id: str

@router.post("/predictLanguageScore")
def predict_language_score(request: TextRequest):
    try:
        scores = predict_all_scores(request.text)
        result = update_assessment(
            assessment_id=request.assessment_id,
            cohesion=scores["cohesion"],
            grammar=scores["grammar"],
            syntax=scores["syntax"],
            overall=scores["overall"],
        )
        return {"status": "success", "predicted_score": scores}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
