from pydantic import BaseModel

class AnswerQuestionRequest(BaseModel):
    question_id: int
    user_id: int
    answered_correctly: int
    time_taken: float
    difficulty_encoded: int