import os
import sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sqlalchemy import create_engine
import requests





def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(
                model,
                para,
                cv=3,
            )
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            # Predict on both training and testing data
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate R² scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store test R² score in report
            report[list(models.keys())[i]] = test_model_score

        return report  # Moved outside the loop

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    



def get_engine():
    user = 'root'
    password = '205011D'
    host = 'localhost'
    port = '3306'
    database = 'exploresDb'
    
    engine = create_engine(f'mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}')
    return engine


from sqlalchemy import create_engine, text  # Add `text` here



from sqlalchemy import  inspect, Table, Column, Integer, String, MetaData


def fetch_and_save_job_titles(source_type="indeed"):
    try:
        if source_type == "indeed":
          

            url = "https://indeed12.p.rapidapi.com/jobs/search"
            querystring = {"query": "software engineer", "sort": "date", "page_id": "1"}
            headers = {
            "X-RapidAPI-Key": "7c77fce846msh2aa9f4dafcc20c4p12d557jsn763e51b0b26b",
            "X-RapidAPI-Host": "indeed12.p.rapidapi.com"
        }
            response = requests.get(url, headers=headers, params=querystring)

            response.raise_for_status()
            data = response.json()
            job_titles = [{"title": job["title"]} for job in data.get("hits", [])]

        elif source_type == "linkedin":
            url = "https://linkedin-data-scraper-api1.p.rapidapi.com/jobs/search"
            querystring = {"keywords": "Software engineer", "sort": "recent"}
            headers = {
                "X-RapidAPI-Key": "7c77fce846msh2aa9f4dafcc20c4p12d557jsn763e51b0b26b",
                "X-RapidAPI-Host": "linkedin-data-scraper-api1.p.rapidapi.com"
            }

            response = requests.get(url, headers=headers, params=querystring)
            response.raise_for_status()
            data = response.json()
            job_list = data.get("data", {}).get("jobs", [])
            job_titles = [{"title": job["job_title"]} for job in job_list]

        else:
            print("❌ Unknown source_type. Use 'indeed' or 'linkedin'.")
            return

        if not job_titles:
            print("No job titles found in API response.")
            return

        # Convert to DataFrame
        df = pd.DataFrame(job_titles)

        # Ensure the column is named 'title'
        df.columns = ['title']

        # Append to DB
        engine = get_engine()
        df.to_sql(name='cleaned_job_titles', con=engine, if_exists='append', index=False)

        print(f"✅ Successfully inserted {len(df)} job titles from {source_type}.")

    except Exception as e:
        print(f"❌ Error fetching data from {source_type} API: {e}")