import uuid
from datetime import datetime
from sqlalchemy import MetaData, Table, insert,select,update,and_, null
from src.utils import get_engine
from src.exception import CustomException
import sys

def create(assessment_id: str,text:str) -> dict:
    try:
        suggestion_id = str(uuid.uuid4())
        created_at = datetime.utcnow()

        engine = get_engine()
        metadata = MetaData()
        metadata.reflect(bind=engine)

        if "suggestions" not in metadata.tables:
            raise Exception("The 'suggestions' table does not exist.")

        assessments_table = metadata.tables["suggestions"]

        insert_stmt = insert(assessments_table).values(
            suggestion_id=suggestion_id,
            assessment_id=assessment_id,
            text=text,
            created_at=created_at,
        )

        with engine.connect() as conn:
            conn.execute(insert_stmt)
            conn.commit()

        return {
            "status": "success",
            "message": "Suggestion created successfully.",
            "suggestion_id": suggestion_id,
        }

    except Exception as e:
        raise CustomException(e, sys)


def createSuggestions(assessment_id: str, suggestions: list) -> dict:
    try:
        engine = get_engine()
        metadata = MetaData()
        metadata.reflect(bind=engine)

        if "suggestions" not in metadata.tables:
            raise Exception("The 'suggestions' table does not exist.")

        suggestions_table = metadata.tables["suggestions"]

        # Prepare a list of insert values
        insert_values = []
        for text in suggestions:
            insert_values.append({
                "suggestion_id": str(uuid.uuid4()),
                "assessment_id": assessment_id,
                "text": text,
                "created_at": datetime.utcnow()
            })

        with engine.connect() as conn:
            conn.execute(insert(suggestions_table), insert_values)
            conn.commit()

        return {
            "status": "success",
            "message": "All suggestions created successfully.",
            "total": len(suggestions)
        }

    except Exception as e:
        raise CustomException(e, sys)
def getSuggestionsByAssessmentId(assessment_id: str) -> dict:
    try:
        engine = get_engine()
        metadata = MetaData()
        metadata.reflect(bind=engine)

        if "suggestions" not in metadata.tables:
            raise Exception("The 'suggestions' table does not exist.")

        suggestions_table = metadata.tables["suggestions"]

        # ✅ Correct usage of select()
        select_stmt = select(
            suggestions_table.c.suggestion_id,
            suggestions_table.c.assessment_id,
            suggestions_table.c.text,
            suggestions_table.c.created_at
        ).where(suggestions_table.c.assessment_id == assessment_id)

        with engine.connect() as conn:
            result = conn.execute(select_stmt)
            suggestions = [dict(row._mapping) for row in result]

        return {
            "status": "success",
            "message": f"{len(suggestions)} suggestion(s) retrieved.",
            "data": suggestions
        }

    except Exception as e:
        raise CustomException(e, sys)