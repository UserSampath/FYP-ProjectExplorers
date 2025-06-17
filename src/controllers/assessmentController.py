import uuid
from datetime import datetime
from sqlalchemy import MetaData, Table, insert,select
from src.utils import get_engine
from src.exception import CustomException
import sys

def create_assessment(user_id: str) -> dict:
    try:
        assessment_id = str(uuid.uuid4())
        created_at = datetime.utcnow()

        engine = get_engine()
        metadata = MetaData()
        metadata.reflect(bind=engine)

        if "assessments" not in metadata.tables:
            raise Exception("The 'assessments' table does not exist.")

        assessments_table = metadata.tables["assessments"]

        insert_stmt = insert(assessments_table).values(
            assessment_id=assessment_id,
            user_id=user_id,
            created_at=created_at,
        )

        with engine.connect() as conn:
            conn.execute(insert_stmt)
            conn.commit()

        return {
            "status": "success",
            "message": "Assessment created successfully.",
            "assessment_id": assessment_id,
        }

    except Exception as e:
        raise CustomException(e, sys)


def get_assessment(assessment_id: str) -> dict:
    try:
        engine = get_engine()
        metadata = MetaData()
        metadata.reflect(bind=engine)

        if "assessments" not in metadata.tables:
            raise Exception("The 'assessments' table does not exist.")

        assessments_table = metadata.tables["assessments"]

        query = select(assessments_table).where(
            assessments_table.c.assessment_id == assessment_id
        )

        with engine.connect() as conn:
            result = conn.execute(query).fetchone()

        if result is None:
            return {
                "status": "error",
                "message": "Assessment not found.",
            }

        return {
            "status": "success",
            "message": "Assessment fetched successfully.",
            "assessment": dict(result._mapping),
        }

    except Exception as e:
        raise CustomException(e, sys)