
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import Table, Column, Integer, String, Boolean, MetaData
from src.utils import get_engine
# File paths
question_path = "notebook/data/questionRecommendation/question­_dataset.csv"
users_path = "notebook/data/questionRecommendation/user.csv"
interaction_path = "notebook/data/questionRecommendation/intaraction_dataset.csv"
job_titles_path = "notebook/data/questionRecommendation/job_titles.csv"

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


engine = get_engine()

metadata = MetaData()

processed_users_table = Table(
    'processed_users',
    metadata,
    Column('user_id', String(255), primary_key=True),  # Keep existing user_id
    Column('expertise_level', String(255)),
    Column('years_of_experience', Integer),
    Column('familiar_technologies', String(255)),
    Column('password', String(255)),
    Column('firstName', String(255)),
    Column('email', String(255)),
    Column('lastName', String(255)),
    Column('otp', String(10)),
    Column('train', Boolean)
)

# Drop and recreate the table
metadata.drop_all(engine, [processed_users_table])
metadata.create_all(engine)

# Add new columns to dfUsers
dfUsers['password'] = ''
dfUsers['firstName'] = ''
dfUsers['lastName'] = ''
dfUsers['email'] = ''
dfUsers['otp'] = ''
dfUsers['train'] = True


# Ensure types are appropriate
dfUsers['user_id'] = dfUsers['user_id'].astype(int)
dfUsers['years_of_experience'] = dfUsers['years_of_experience'].fillna(0).astype(int)
dfUsers['train'] = dfUsers['train'].astype(bool)



# Create cleaned_job_titles table with auto-increment ID
job_titles_table = Table(
    'cleaned_job_titles',
    metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('title', String(255))
)

# Drop if exists and create new
metadata.drop_all(engine, [job_titles_table])
metadata.create_all(engine)

# Write data (excluding ID so it auto-increments)
dfJobTitles.columns = ['title']  # Ensure correct column name
dfJobTitles.to_sql(name='cleaned_job_titles', con=engine, if_exists='append', index=False)



 #Save processed files in db
dfQuestion.to_sql(name='processed_question', con=engine, if_exists='replace', index=False)
dfUsers.to_sql(name='processed_users', con=engine, if_exists='append', index=False)
dfInteractions.to_sql(name='processed_interactions', con=engine, if_exists='replace', index=False)
# dfJobTitles.to_sql(name='cleaned_job_titles', con=engine, if_exists='replace', index=False) 


# # Save processed files
# dfQuestion.to_csv(f"{ARTIFACT_DIR}/processed_question.csv", index=False)
# dfUsers.to_csv(f"{ARTIFACT_DIR}/processed_users.csv", index=False)
# dfInteractions.to_csv(f"{ARTIFACT_DIR}/processed_interactions.csv", index=False)
# dfJobTitles.to_csv(f"{ARTIFACT_DIR}/cleaned_job_titles.csv", index=False)