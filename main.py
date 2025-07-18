from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from routers import jobTitles, question, audio, language,user,assessment,languageProficiencyAssessment,videoInterviewAssessment,videoEmotion


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

app.include_router(question.router, prefix="/questions", tags=["Personalized Questions"])
app.include_router(audio.router, prefix="/audio", tags=["Language Proficiency"])
app.include_router(language.router, prefix="/language", tags=["Language Proficiency"])
app.include_router(jobTitles.router, prefix="/jobs", tags=["Personalized Questions"])
app.include_router(user.router, prefix="/user", tags=["User"])
app.include_router(assessment.router, prefix="/assessment", tags=["Personalized Questions"])
app.include_router(languageProficiencyAssessment.router, prefix="/languageProficiencyAssessment", tags=["Language Proficiency"])
app.include_router(videoEmotion.router, prefix="/videoEmotion", tags=["Video Emotion Analysis"])
app.include_router(videoInterviewAssessment.router, prefix="/videoInterviewAssessment", tags=["Video Emotion Analysis"])






