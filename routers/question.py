from fastapi import APIRouter, HTTPException,Depends
from src.pipeline.questionRecommendation.recommendQuestion import hybrid_recommendations
from src.schemas.schemas import AnswerQuestionRequest,APIResponse,AnswerQuestionsRequest
from src.controllers.questionController import answer_question,answer_questions,question_by_user
from src.controllers.assessmentController import update_assessment
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

@router.post("/answerQuestions")
def save_answers(req: AnswerQuestionsRequest,  user_id: int = Depends(get_current_user)):
    try:
        print(user_id)
        # Call your controller function
        result = answer_questions(
            user_id=user_id,
            assessment_id=req.assessment_id,
            questions=req.questions
        )

        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
        
        result2 = update_assessment(
            assessment_id=req.assessment_id,
            correct=result["correct_answers"],
            question_count=result["total_questions"],
        )

        if result2.get("status") == "error":
            raise_custom_error(404, result.get("message", "Update assessment failed"))

        return {
            "status": "success",
            "success": True,
            "message": result.get("message", "Answers saved successfully."),
            "data": result.get("data", {})
        }

    except HTTPException as he:
        raise he

    except Exception as e:
        raise_custom_error(500, f"Internal Server Error: {str(e)}")

@router.get("/getByUser")
def getQuestionsByUser(user_id: int = Depends(get_current_user)):
    try:
        result = question_by_user(
            user_id=user_id,
        )

        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
        
        return {
            "status": "success",
            "success": True,
            "message": result.get("message", "Answers saved successfully."),
            "data": result.get("data")
        }

    except HTTPException as he:
        raise he

    except Exception as e:
        raise_custom_error(500, f"Internal Server Error: {str(e)}")