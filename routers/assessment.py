from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from src.controllers.assessmentController import create_assessment,get_assessment
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