import pandas as pd
from sqlalchemy import insert,text
from src.utils import get_engine
from typing import List
from src.schemas.schemas import QuestionAnswer
from datetime import datetime,timezone
from src.pipeline.questionRecommendation.recommendQuestionGraphBased import add_new_interaction
def answer_question(question_id: int, user_id: int, answered_correctly: bool,
                    time_taken: float, difficulty_encoded: float):
    try:

        max_time = 30  
        min_time = 3   
        timeTaken_minmax = (time_taken - min_time) / (max_time - min_time)
        timeTaken_minmax = max(0, min(timeTaken_minmax, 1))  
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

def answer_questions(user_id: str, assessment_id: str, questions: List[QuestionAnswer]) -> dict:
    try:
        print("Questions",questions)
        processed_data = []

        max_time = 30
        min_time = 2

        difficulty_map = {'Easy': 1, 'Medium': 2, 'Hard': 3}

        engine = get_engine()
        question_ids = [q.question_id for q in questions]
        placeholders = ','.join([':id' + str(i) for i in range(len(question_ids))])
        query = f"""
            SELECT question_id, correct_option, difficulty_level 
            FROM processed_question 
            WHERE question_id IN ({placeholders})
        """
        query_params = {f'id{i}': qid for i, qid in enumerate(question_ids)}

        with engine.connect() as conn:
            result = conn.execute(text(query), query_params).mappings()  # ← add .mappings()
            question_info = {
            row['question_id']: {
            'correct_option': row['correct_option'],
            'difficulty_encoded': difficulty_map.get(row['difficulty_level'], 0)
             }
                 for row in result
            }


        for q in questions:
            q_info = question_info.get(q.question_id)

            if not q_info:
                continue  # Skip if question not found

            selected_option = q.selected_option
            correct_option = q_info['correct_option']
            difficulty_encoded = q_info['difficulty_encoded']

            # print("correct 🥲",selected_option,correct_option) 
            is_correct = int(selected_option == correct_option)

            timeTaken_minmax = (q.time_taken - min_time) / (max_time - min_time)
            timeTaken_minmax = max(0, min(timeTaken_minmax, 1))

            row = {
                "question_id": q.question_id,
                "user_id": user_id,
                "answered_correctly": is_correct,
                "time_taken": q.time_taken,
                "timeTaken_minmax": timeTaken_minmax,
                "difficulty_encoded": difficulty_encoded,
                "selected_option": selected_option,
                "assessment_id": assessment_id,
                "created_at": datetime.now(timezone.utc)
            }

            new_interaction = {"user_id": user_id,
                               "question_id": q.question_id,
                               "answered_correctly": is_correct,
                               "created_at": datetime.now(timezone.utc)
                               }
            add_new_interaction(new_interaction)
            

            processed_data.append(row)

        if processed_data:
            df = pd.DataFrame(processed_data)
            df.to_sql('processed_interactions', engine, if_exists='append', index=False)

        correct_count = sum(row['answered_correctly'] for row in processed_data)
        total_count = len(processed_data)

        return {
            "status": "success",
            "message": f"{total_count} answers saved successfully",
            "correct_answers": correct_count,
            "total_questions": total_count,
            "data": processed_data
        }

    except Exception as e:
        print("Error in answer_questions:", e)
        return {
            "status": "error",
            "message": str(e)
        }
    
def question_by_user(user_id: str) -> dict:
    try:
        engine = get_engine()
        query = """
            SELECT 
                i.question_id,
                q.question,
                q.option_a,
                q.option_b,
                q.option_c,
                q.option_d,
                q.category,
                i.selected_option,
                q.correct_option,
                i.answered_correctly,
                i.time_taken,
                i.assessment_id,
                i.created_at
            FROM processed_interactions i
            JOIN processed_question q ON i.question_id = q.question_id
            WHERE i.user_id = :user_id
            ORDER BY i.created_at DESC
            LIMIT 60
        """

        with engine.connect() as conn:
            result = conn.execute(text(query), {"user_id": user_id}).mappings()
            data = [dict(row) for row in result]

        return {
            "status": "success",
            "message": f"Retrieved {len(data)} answered questions for user.",
            "data": data
        }

    except Exception as e:
        print("Error in question_by_user:", e)
        return {
            "status": "error",
            "message": str(e)
        }
    
def question_by_assessment(assessment_id: str) -> dict:
    try:
        engine = get_engine()
        query = """
            SELECT 
                i.question_id,
                q.question,
                q.option_a,
                q.option_b,
                q.option_c,
                q.option_d,
                q.category,
                i.selected_option,
                q.correct_option,
                i.answered_correctly,
                i.time_taken,
                i.assessment_id,
                i.created_at
            FROM processed_interactions i
            JOIN processed_question q ON i.question_id = q.question_id
            WHERE i.assessment_id = :assessment_id
        """

        with engine.connect() as conn:
            result = conn.execute(text(query), {"assessment_id": assessment_id}).mappings()
            data = [dict(row) for row in result]

        return {
            "status": "success",
            "message": f"Retrieved {len(data)} answered questions for assessment {assessment_id}.",
            "data": data
        }

    except Exception as e:
        print("Error in question_by_assessment:", e)
        return {
            "status": "error",
            "message": str(e)
        }