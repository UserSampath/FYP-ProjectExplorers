from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel,Field
from src.controllers.videoInterviewAssessmentController import create_assessment,get_assessment,getUserLastPerformance,update_assessment
from src.middleware.findUser import get_current_user
from src.exception import raise_custom_error
from typing import List, Optional
from datetime import datetime
router = APIRouter()

# Request schema for creating assessment (only user_id from token, so no body needed)
class AssessmentCreateResponse(BaseModel):
    status: str
    success: bool
    message: str
    data: dict

class AssessmentModel(BaseModel):
    assessment_id: str
    user_id: str
    topic: str
    average_confidence: Optional[float]
    stability_score: Optional[float]
    final_confidence: Optional[float]
    penalty_adjusted_confidence: Optional[float]
    confidence_label: Optional[str]
    created_at: datetime

class AssessmentsGetResponse(BaseModel):
    status: str
    success: bool
    message: str
    data: List[AssessmentModel] 

class CreateAssessmentRequest(BaseModel):
    topic: str

@router.post("/create", response_model=AssessmentCreateResponse)
def create_assessment_endpoint(req:CreateAssessmentRequest,user_id: str = Depends(get_current_user)):
    try:
        result = create_assessment(user_id=user_id,topic=req.topic)

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


@router.get("/getUserLastPerformance", response_model=AssessmentsGetResponse)
def fetch_assessment(user_id: str = Depends(get_current_user)):
    try:
        result = getUserLastPerformance(user_id)
        if result["status"] == "error":
            raise_custom_error(404, result["message"])
        return {
            "status": "success",
            "success": True,
            "message": result["message"],
            "data": result["assessments"]
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise_custom_error(500, f"Internal Server Error: {str(e)}")
class AssessmentUpdateRequest(BaseModel):
    average_confidence: float
    stability_score: float
    final_confidence: float
    penalty_adjusted_confidence: float
    confidence_label:str


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
            average_confidence=payload.average_confidence,
            stability_score=payload.stability_score,
            final_confidence=payload.final_confidence,
            penalty_adjusted_confidence=payload.penalty_adjusted_confidence,
            confidence_label=payload.confidence_label
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