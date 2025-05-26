from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from sklearn.preprocessing import StandardScaler


from src.pipeline.Question_Recommendation.recammandQuestion import (
    hybrid_recommendations
)

# ...existing code...
app = Flask(__name__)
# Initialize CORS with default options
CORS(app)

# Remove the duplicate Flask app initialization
# app = Flask(__name__)  # Remove this line


@app.route("/")
def index():
    return "API is working"






@app.route("/questionrecommendation", methods=["POST"])
def question_recommendation():
    try:
        data_json = request.get_json()
        print("Received data for question recommendation:", data_json)

        user_id = int(data_json["user_id"])
        num_questions = int(data_json.get("num_questions", 5))  # Optional: allow client to set number of questions

        recommendations_df = hybrid_recommendations(user_id, num_questions=num_questions)

        result = recommendations_df[[
            "question_id", "question", "topic", "tags", "difficulty_level"
        ]].to_dict(orient="records")

        return jsonify({
            "status": "success",
            "user_id": user_id,
            "recommended_questions": result
        })

    except Exception as e:
        print("Error during question recommendation:", e)
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
