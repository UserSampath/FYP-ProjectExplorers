import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds

predicted_df = None

def update_collab_model(dfInteractions):
    global predicted_df
    dfInteractions['weighted_score'] = (
        dfInteractions['answered_correctly'] +
        dfInteractions['timeTaken_minmax'] +
        dfInteractions['difficulty_encoded']
    ) / 3

    # Pivot to create the user-question matrix
    interaction_matrix = dfInteractions.pivot_table(
        index='user_id',
        columns='question_id',
        values='weighted_score',
        aggfunc='mean'
    ).fillna(0)
    print("\n dfInteractions:\n", dfInteractions.round(2))
    interaction_np = interaction_matrix.values

    # Apply SVD
    k = min(2, min(interaction_np.shape) - 1)  # Ensure valid k
    U, sigma, Vt = svds(interaction_np, k=k)
    sigma = np.diag(sigma)
    predicted = np.dot(np.dot(U, sigma), Vt)
    predicted_df = pd.DataFrame(predicted, index=interaction_matrix.index, columns=interaction_matrix.columns)
    print("\n Predicted Rating Matrix (predicted_df):\n", predicted_df.round(2))


def get_answered_questions(dfInteractions, user_id):
    return dfInteractions[dfInteractions['user_id'] == user_id]['question_id'].unique().tolist()


def recommend_questions_collab(user_id, dfInteractions, n=5):
    if user_id not in predicted_df.index:
        return []
    answered = get_answered_questions(dfInteractions, user_id)
    ranked = predicted_df.loc[user_id].sort_values(ascending=False)


    unseen_recommendations = [qid for qid in ranked.index if qid not in answered][:n]

 
    print(f"Recommended Questions for {user_id}: {unseen_recommendations}")

    return unseen_recommendations


if __name__ == "__main__":
    # Sample interaction data
    data = {
        'user_id': [1, 1, 1, 2, 2, 3, 3, 1, 2, 3, 2],
        'question_id': [101, 102, 103, 101, 104, 102, 105, 106, 107, 108, 109],
        'answered_correctly': [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0],
        'timeTaken_minmax': [0.8, 0.5, 0.7, 0.6, 0.4, 0.9, 0.7, 0.9, 0.3, 0.8, 0.2],
        'difficulty_encoded': [1, 0, 2, 1, 2, 1, 1, 1, 2, 0, 2],
    }

    dfInteractions = pd.DataFrame(data)

    # Train collaborative model
    update_collab_model(dfInteractions)

    # Display predicted ratings
    # print("\nPredicted rating matrix:")
    # print(predicted_df.round(2))

    # Recommend top 3 questions for user 1
    recommended = recommend_questions_collab(user_id=1, dfInteractions=dfInteractions, n=3)
    # print(f"\nRecommended questions for user 1: {recommended}")