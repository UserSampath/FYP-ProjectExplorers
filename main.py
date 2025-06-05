from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from routers import jobTitles, question, audio, language


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

app.include_router(question.router, prefix="/questions", tags=["Questions"])
app.include_router(audio.router, prefix="/audio", tags=["Audio Analysis"])
app.include_router(language.router, prefix="/language", tags=["Language Proficiency"])
app.include_router(jobTitles.router, prefix="/jobs", tags=["Job Titles"])
