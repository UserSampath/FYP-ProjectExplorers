
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# File paths
question_path = "notebook/data/questionRecommendation/question­_dataset.csv"
users_path = "notebook/data/questionRecommendation/user.csv"
interaction_path = "notebook/data/questionRecommendation/intaraction_dataset.csv"
job_titles_path = "notebook/data/questionRecommendation/jobTitles.csv"

# Load datasets
dfQuestion = pd.read_csv(question_path)
dfUsers = pd.read_csv(users_path)
dfInteractions = pd.read_csv(interaction_path)
dfJobTitles = pd.read_csv(job_titles_path)

# Clean and preprocess interaction data
dfInteractions['time_taken'] = pd.to_numeric(dfInteractions['time_taken'], errors='coerce')
dfInteractions['time_taken'].fillna(dfInteractions['time_taken'].median(), inplace=True)

scaler = MinMaxScaler()
dfInteractions['timeTaken_minmax'] = dfInteractions.groupby('user_id')['time_taken'].transform(
    lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten() if len(x) > 1 else 0
)

dfInteractions['answerd_correctly'] = dfInteractions['answerd_correctly'].map({'Yes': 1, 'No': 0})

# Encode difficulty
difficulty_map = {'Easy': 1, 'Medium': 2, 'Hard': 3}
dfQuestion['difficulty_encoded'] = dfQuestion['difficulty_level'].map(difficulty_map)
dfInteractions = dfInteractions.merge(dfQuestion[['question_id', 'difficulty_encoded']], on='question_id', how='left')


ARTIFACT_DIR = "artifact/question_recommendation"

# Save processed files
dfQuestion.to_csv(f"{ARTIFACT_DIR}/processed_question.csv", index=False)
dfUsers.to_csv(f"{ARTIFACT_DIR}/processed_users.csv", index=False)
dfInteractions.to_csv(f"{ARTIFACT_DIR}/processed_interactions.csv", index=False)
dfJobTitles.to_csv(f"{ARTIFACT_DIR}/cleaned_job_titles.csv", index=False)