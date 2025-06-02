import pandas as pd
from sqlalchemy import insert
from src.utils import get_engine

# Load any necessary scalers, encoders if needed

def answer_question(question_id: int, user_id: int, answered_correctly: bool,
                    time_taken: float, difficulty_encoded: float):
    try:
        # Compute timeTaken_minmax - Optionally use global MinMax, bucket logic, or fixed scaling
        # For now, here's an example using fixed scaling
        max_time = 30  # e.g., 2 minutes
        min_time = 3   # e.g., 1 second
        timeTaken_minmax = (time_taken - min_time) / (max_time - min_time)
        timeTaken_minmax = max(0, min(timeTaken_minmax, 1))  # clip between 0 and 1

        # Save data to DB
        data = {
            "question_id": question_id,
            "user_id": user_id,
            "answered_correctly": int(answered_correctly),
            "time_taken": time_taken,
            "timeTaken_minmax": timeTaken_minmax,
            "difficulty_encoded": difficulty_encoded
        }

        df = pd.DataFrame([data])

        engine = get_engine()
        df.to_sql('processed_interactions', engine, if_exists='append', index=False)

        return {
            "status": "success",
            "message": "Answer saved successfully",
            "data": data
        }

    except Exception as e:
        print("Error in answer_question:", e)
        return {
            "status": "error",
            "message": str(e)
        }
