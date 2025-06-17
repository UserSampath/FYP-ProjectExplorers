from fastapi import APIRouter, HTTPException,Depends
from src.pipeline.questionRecommendation.recommendQuestion import hybrid_recommendations
from src.schemas.schemas import AnswerQuestionRequest,APIResponse
from src.controllers.questionController import answer_question
from pydantic import BaseModel
from src.middleware.findUser import get_current_user
from src.exception import raise_custom_error
router = APIRouter()

class RecommendationRequest(BaseModel):
    num_questions: int = 5

@router.post("/generateQuestions",response_model=APIResponse)
def question_recommendation( req: RecommendationRequest,user_id: int = Depends(get_current_user)):
    try:
        recommendations_df = hybrid_recommendations(user_id, req.num_questions)
        result = recommendations_df[["question_id", "question", "topic", "tags", "difficulty_level","option_A","option_B","option_C","option_C","option_D"]].to_dict(orient="records")
        return {"status":"200", "success":True, "message":"Success getting questions" ,"data":{
"user_id": user_id, "recommended_questions": result
        }}
    except HTTPException as he:
        raise he

    except Exception as e:
        raise_custom_error(500, f"Internal Server Error: {str(e)}")

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
    except HTTPException as he:
        raise he

    except Exception as e:
        raise_custom_error(500, f"Internal Server Error: {str(e)}")
