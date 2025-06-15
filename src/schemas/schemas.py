from pydantic import BaseModel
from typing import Optional,Any

class AnswerQuestionRequest(BaseModel):
    question_id: int
    user_id: int
    answered_correctly: int
    time_taken: float
    difficulty_encoded: int

class RegisterRequest(BaseModel):
    fullName:str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str

class UpdateUserRequest(BaseModel):
    fullName: Optional[str] = None
    email: Optional[str] = None
    familiar_technologies: Optional[str] = None
    years_of_experience: Optional[int] = None
    expertise_level: Optional[str] = None


class APIResponse(BaseModel):
    status: str 
    success: bool
    message: str
    data: Any 