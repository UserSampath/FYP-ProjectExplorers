from fastapi import APIRouter, HTTPException
from src.pipeline.Question_Recommendation.recommendQuestion import hybrid_recommendations
from src.schemas.schemas import AnswerQuestionRequest
from src.controllers.question_controller import answer_question
from pydantic import BaseModel

router = APIRouter()

class RecommendationRequest(BaseModel):
    user_id: int
    num_questions: int = 5

@router.post("/generateQuestions")
def question_recommendation(req: RecommendationRequest):
    try:
        recommendations_df = hybrid_recommendations(req.user_id, req.num_questions)
        result = recommendations_df[["question_id", "question", "topic", "tags", "difficulty_level"]].to_dict(orient="records")
        return {"status": "success", "user_id": req.user_id, "recommended_questions": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/answerQuestion")
def save_answer(req: AnswerQuestionRequest):
    try:
        result = answer_question(
            question_id=req.question_id,
            user_id=req.user_id,
            answered_correctly=req.answered_correctly,
            time_taken=req.time_taken,
            difficulty_encoded=req.difficulty_encoded
        )
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
