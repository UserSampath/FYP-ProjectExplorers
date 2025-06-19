from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel,Field
from src.controllers.assessmentController import create_assessment,get_assessment,update_assessment
from src.middleware.findUser import get_current_user
from src.exception import raise_custom_error

router = APIRouter()

# Request schema for creating assessment (only user_id from token, so no body needed)
class AssessmentCreateResponse(BaseModel):
    status: str
    success: bool
    message: str
    data: dict

@router.post("/create", response_model=AssessmentCreateResponse)
def create_assessment_endpoint(user_id: str = Depends(get_current_user)):
    """
    Create a new assessment for the logged-in user.
    """
    try:
        result = create_assessment(user_id=user_id)

        if result.get("status") == "error":
            raise_custom_error(400, result.get("message", "Failed to create assessment"))

        return {
            "status": "success",
            "success": True,
            "message": "Assessment created successfully",
            "data": {"assessment_id": result.get("assessment_id")},
        }

    except HTTPException as he:
        raise he

    except Exception as e:
        raise_custom_error(500, f"Internal Server Error: {str(e)}")


@router.get("/get/{assessment_id}")
def fetch_assessment(assessment_id: str):
    try:
        result = get_assessment(assessment_id)
        if result["status"] == "error":
            raise_custom_error(404, result["message"])
        return {
            "status": "success",
            "success": True,
            "message": result["message"],
            "data": result["assessment"]
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise_custom_error(500, f"Internal Server Error: {str(e)}")

class AssessmentUpdateRequest(BaseModel):
    correct: int = Field(..., ge=0, description="Number of correct answers")
    question_count: int = Field(..., ge=0, description="Total number of questions")

# Response model for assessment operations
class AssessmentResponse(BaseModel):
    status: str
    success: bool
    message: str
    data: dict = None


@router.put("/update/{assessment_id}", response_model=AssessmentResponse)
def update_assessment_endpoint(
    assessment_id: str,
    payload: AssessmentUpdateRequest,
):
    """
    Update an existing assessment's correct and question_count values.
    """
    try:
        result = update_assessment(
            assessment_id=assessment_id,
            correct=payload.correct,
            question_count=payload.question_count,
        )

        if result.get("status") == "error":
            raise_custom_error(404, result.get("message", "Update failed"))

        return {
            "status": "success",
            "success": True,
            "message": result["message"],
            "data": {"assessment_id": result["assessment_id"]},
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise_custom_error(500, f"Internal Server Error: {str(e)}")