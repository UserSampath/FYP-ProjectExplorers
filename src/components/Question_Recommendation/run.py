
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.utils import get_engine
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
dfInteractions['time_taken'] = dfInteractions['time_taken'].fillna(dfInteractions['time_taken'].median())


scaler = MinMaxScaler()
dfInteractions['timeTaken_minmax'] = scaler.fit_transform(dfInteractions[['time_taken']])

dfInteractions['answered_correctly'] = dfInteractions['answered_correctly'].map({'Yes': 1, 'No': 0})

# Encode difficulty
difficulty_map = {'Easy': 1, 'Medium': 2, 'Hard': 3}
dfQuestion['difficulty_encoded'] = dfQuestion['difficulty_level'].map(difficulty_map)
dfInteractions = dfInteractions.merge(dfQuestion[['question_id', 'difficulty_encoded']], on='question_id', how='left')


ARTIFACT_DIR = "artifact/question_recommendation"

engine = get_engine()

 #Save processed files in db
dfQuestion.to_sql(name='processed_question', con=engine, if_exists='replace', index=False)
dfUsers.to_sql(name='processed_users', con=engine, if_exists='replace', index=False)
dfInteractions.to_sql(name='processed_interactions', con=engine, if_exists='replace', index=False)
dfJobTitles.to_sql(name='cleaned_job_titles', con=engine, if_exists='replace', index=False) 


# # Save processed files
# dfQuestion.to_csv(f"{ARTIFACT_DIR}/processed_question.csv", index=False)
# dfUsers.to_csv(f"{ARTIFACT_DIR}/processed_users.csv", index=False)
# dfInteractions.to_csv(f"{ARTIFACT_DIR}/processed_interactions.csv", index=False)
# dfJobTitles.to_csv(f"{ARTIFACT_DIR}/cleaned_job_titles.csv", index=False)