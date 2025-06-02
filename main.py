from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from sklearn.preprocessing import StandardScaler
from src.pipeline.Question_Recommendation.recommendQuestion import hybrid_recommendations
from src.schemas.schemas import AnswerQuestionRequest
from src.controllers.question_controller import answer_question


# question
# Define the request body model
class RecommendationRequest(BaseModel):
    user_id: int
    num_questions: int = 5  # default to 5

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def index():
    return {"message": "API is working"}

@app.post("/generateQuestions")
def question_recommendation(req: RecommendationRequest):
    try:
        print("Received data for question recommendation:", req)

        user_id = req.user_id
        num_questions = req.num_questions

        recommendations_df = hybrid_recommendations(user_id, num_questions=num_questions)

        result = recommendations_df[[
            "question_id", "question", "topic", "tags", "difficulty_level"
        ]].to_dict(orient="records")

        return {
            "status": "success",
            "user_id": user_id,
            "recommended_questions": result
        }

    except Exception as e:
        print("Error during question recommendation:", e)
        raise HTTPException(status_code=500, detail=str(e))
    

    
@app.post("/answerQuestion")
def save_answer(req: AnswerQuestionRequest):
    try:
        print("Saving answer:", req)

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
        print("Error in /answerQuestion:", e)
        raise HTTPException(status_code=500, detail=str(e))