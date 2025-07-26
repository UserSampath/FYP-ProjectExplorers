import uuid
from datetime import datetime
from sqlalchemy import MetaData, Table, insert,select,update,and_, null
from src.utils import get_engine
from src.exception import CustomException
import sys

def create_assessment(user_id: str,topic:str) -> dict:
    try:
        assessment_id = str(uuid.uuid4())
        created_at = datetime.utcnow()

        engine = get_engine()
        metadata = MetaData()
        metadata.reflect(bind=engine)

        if "videointerviewassessments" not in metadata.tables:
            raise Exception("The 'videointerviewassessments' table does not exist.")

        assessments_table = metadata.tables["videointerviewassessments"]

        insert_stmt = insert(assessments_table).values(
            assessment_id=assessment_id,
            user_id=user_id,
            topic=topic,
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

        if "videointerviewassessments" not in metadata.tables:
            raise Exception("The 'videointerviewassessments' table does not exist.")

        assessments_table = metadata.tables["videointerviewassessments"]

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
    
def getUserLastPerformance(userId: str) -> dict:
    try:

        print(f"Fetching last performance for user: {userId}")
        engine = get_engine()
        metadata = MetaData()
        metadata.reflect(bind=engine)

        if "videointerviewassessments" not in metadata.tables:
            raise Exception("The 'videointerviewassessments' table does not exist.")

        assessments_table = metadata.tables["videointerviewassessments"]

        query = (
            select(assessments_table)
            .where(
            and_(
            assessments_table.c.user_id == userId,
            assessments_table.c.final_confidence.isnot(null())
            )
    )
    .order_by(assessments_table.c.created_at.desc())
    .limit(5)
)

        with engine.connect() as conn:
            results = conn.execute(query).fetchall()
        assessments = [dict(row._mapping) for row in results]

        return {
            "status": "success",
            "message": "Assessments fetched successfully.",
            "assessments": assessments,
        }

    except Exception as e:
        print(f"Error fetching last performance: {str(e)}")
        raise CustomException(e, sys)

def update_assessment(
    assessment_id: str,
    average_confidence: float,
    stability_score: float,
    final_confidence: float,
    penalty_adjusted_confidence: float,
    confidence_label: str,
    audio_emotion:str,
    emotion_counts: dict = None,
) -> dict:
    try:
        engine = get_engine()
        metadata = MetaData()
        metadata.reflect(bind=engine)

        if "videointerviewassessments" not in metadata.tables:
            raise Exception("The 'videointerviewassessments' table does not exist.")

        assessments_table = metadata.tables["videointerviewassessments"]

        update_values = {
            "average_confidence": average_confidence,
            "stability_score": stability_score,
            "final_confidence": final_confidence,
            "penalty_adjusted_confidence": penalty_adjusted_confidence,
            "confidence_label": confidence_label,
            "audio_emotion":audio_emotion
        }

        # Add emotion counts if provided
        if emotion_counts:
            for emotion in ["happy", "surprise", "neutral", "calm", "sad", "fear", "angry", "disgust"]:
                update_values[emotion] = emotion_counts.get(emotion, 0)

        update_stmt = (
            update(assessments_table)
            .where(assessments_table.c.assessment_id == assessment_id)
            .values(**update_values)
        )

        with engine.connect() as conn:
            result = conn.execute(update_stmt)
            conn.commit()

        if result.rowcount == 0:
            return {
                "status": "error",
                "message": "Assessment not found or no update performed.",
            }

        return {
            "status": "success",
            "message": "Assessment updated successfully.",
            "assessment_id": assessment_id,
        }

    except Exception as e:
        raise CustomException(e, sys)
