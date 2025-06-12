from pydantic import BaseModel
from typing import Optional

class AnswerQuestionRequest(BaseModel):
    question_id: int
    user_id: int
    answered_correctly: int
    time_taken: float
    difficulty_encoded: int

class RegisterRequest(BaseModel):
    firstName: str
    lastName: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str

class UpdateUserRequest(BaseModel):
    firstName: Optional[str] = None
    lastName: Optional[str] = None
    email: Optional[str] = None
    password: Optional[str] = None