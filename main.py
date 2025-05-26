from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from sklearn.preprocessing import StandardScaler
from src.pipeline.Question_Recommendation.recammandQuestion import hybrid_recommendations

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

@app.post("/questionrecommendation")
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
