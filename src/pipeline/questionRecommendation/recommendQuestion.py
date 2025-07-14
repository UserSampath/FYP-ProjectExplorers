import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from src.utils import get_engine

# Global Models
predicted_df = None
vectorizer = None
tfidf_matrix = None
recommender = None

# Load technical keywords
csv_path = Path("notebook/data/questionRecommendation/technical_words.csv")
df_keywords = pd.read_csv(csv_path)
TECH_KEYWORDS = df_keywords['technical_word'].dropna().str.strip().str.lower().unique().tolist()


# ==================== Data Load ====================

def load_data():
    engine = get_engine()
    dfQuestion = pd.read_sql("SELECT * FROM processed_question", engine)
    dfUsers = pd.read_sql("SELECT * FROM processed_users", engine)
    dfInteractions = pd.read_sql("SELECT * FROM processed_interactions", engine)
    return dfQuestion, dfUsers, dfInteractions


# ==================== Preprocessing ====================

def match_technology(row):
    return any(
        any(tech in str(row[col]) for tech in TECH_KEYWORDS)
        for col in ['question', 'topic', 'tags']
    )

def update_tfidf_model(dfQuestion):
    global vectorizer, tfidf_matrix
    dfQuestion.fillna('', inplace=True)
    dfQuestion['combined'] = dfQuestion[['topic', 'tags', 'question']].agg(' '.join, axis=1)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(dfQuestion['combined'])


def update_collab_model(dfInteractions):
    global predicted_df
    dfInteractions['weighted_score'] = (
        dfInteractions['answered_correctly'] +
        dfInteractions['timeTaken_minmax'] +
        dfInteractions['difficulty_encoded']
    ) / 3

    interaction_matrix = dfInteractions.pivot_table(
        index='user_id',
        columns='question_id',
        values='weighted_score',
        aggfunc='mean'
    ).fillna(0)

    interaction_np = interaction_matrix.values
    U, sigma, Vt = svds(interaction_np, k=5)
    sigma = np.diag(sigma)
    predicted = np.dot(np.dot(U, sigma), Vt)
    predicted_df = pd.DataFrame(predicted, index=interaction_matrix.index, columns=interaction_matrix.columns)


# ==================== Recommenders ====================

def get_answered_questions(dfInteractions, user_id):
    return set(dfInteractions[dfInteractions['user_id'] == user_id]['question_id'].tolist())


def recommend_questions_collab(user_id, dfInteractions, n=5):
    if user_id not in predicted_df.index:
        return []
    answered = get_answered_questions(dfInteractions, user_id)
    ranked = predicted_df.loc[user_id].sort_values(ascending=False)
    return [qid for qid in ranked.index if qid not in answered][:n]


def recommend_questions_content(user_id, dfUsers, dfQuestion, dfInteractions, n=5):
    user = dfUsers[dfUsers['user_id'] == user_id]
    if user.empty:
        return []
    answered = get_answered_questions(dfInteractions, user_id)
    prefs = ' '.join(user['familiar_technologies'].astype(str).tolist())
    user_vec = vectorizer.transform([prefs])
    scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
    top_idx = scores.argsort()[::-1]
    recs = [int(dfQuestion.iloc[i]['question_id']) for i in top_idx if dfQuestion.iloc[i]['question_id'] not in answered]
    return recs[:n]


class QuestionBanditRecommender:
    def __init__(self, dfQuestion, dfUsers, dfInteractions):
        self.qdf = dfQuestion
        self.udf = dfUsers
        self.idf = dfInteractions
        self.successes = {qid: 0 for qid in dfQuestion['question_id']}
        self.attempts = {qid: 1 for qid in dfQuestion['question_id']}
        for _, row in dfInteractions.iterrows():
            qid = row['question_id']
            self.attempts[qid] += 1
            if row['answered_correctly'] == 1:
                t = row['time_taken']
                self.successes[qid] += 1 if t <= 10 else 0.8 if t <= 20 else 0.5

    def ucb_score(self, qid, total):
        mean = self.successes[qid] / self.attempts[qid]
        return mean + np.sqrt((2 * np.log(total)) / self.attempts[qid])

    def recommend(self, user_id, top_n=5):
        user = self.udf[self.udf['user_id'] == user_id]
        if user.empty:
            return pd.DataFrame(columns=['question_id'])
        user = user.iloc[0]
        user_techs = [t.strip().lower() for t in str(user['familiar_technologies']).split(',')]
        level = str(user['expertise_level']).lower()
        answered = set(map(int, self.idf[self.idf['user_id'] == user_id]['question_id'].tolist()))
        candidates = self.qdf[~self.qdf['question_id'].isin(answered)]
        total_attempts = sum(self.attempts.values())
        scored = []
        for _, q in candidates.iterrows():
            qid = q['question_id']
            score = self.ucb_score(qid, total_attempts)
            tags = [t.strip().lower() for t in str(q['tags']).split(',')]
            topic_match = any(tech in tags for tech in user_techs)
            difficulty_match = q['difficulty_level'].strip().lower() == level
            if topic_match:
                score *= 1.2
            if difficulty_match:
                score *= 1.1
            scored.append((qid, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = [qid for qid, _ in scored[:top_n]]
        return self.qdf[self.qdf['question_id'].isin(top)]


def recommend_questions_job_title_only(dfQuestion, dfInteractions, user_id, n=20):
    answered = get_answered_questions(dfInteractions, user_id)

    def count_keyword_matches(text):
        text = str(text).lower()
        return sum(1 for keyword in TECH_KEYWORDS if keyword in text)

    # Filter unanswered questions with keyword match
    job_related = dfQuestion[dfQuestion['question_id'].isin(
        dfQuestion[dfQuestion['tech_keyword_match'] == 1]['question_id']
    ) & (~dfQuestion['question_id'].isin(answered))].copy()

    # Compute relevance score
    job_related['match_score'] = job_related.apply(
        lambda row: (
            count_keyword_matches(row['question']) +
            count_keyword_matches(row['tags']) +
            count_keyword_matches(row['topic'])
        ),
        axis=1
    )

    # Sort by highest score and return top-n
    job_related = job_related.sort_values(by='match_score', ascending=False)
    return job_related.head(n)['question_id'].tolist()



# ==================== Hybrid Recommender ====================

def hybrid_recommendations(user_id, num_questions=5, alpha=0.35, beta=0.25, gamma=0.25, delta=0.15):
    global recommender

    # Load fresh data
    dfQuestion, dfUsers, dfInteractions = load_data()

    # Preprocess
    str_cols = dfQuestion.select_dtypes(include=['object']).columns
    dfQuestion[str_cols] = dfQuestion[str_cols].apply(lambda col: col.str.lower())
    dfQuestion['tech_keyword_match'] = dfQuestion.apply(match_technology, axis=1).astype(int)

    # Update models
    update_collab_model(dfInteractions)
    update_tfidf_model(dfQuestion)
    recommender = QuestionBanditRecommender(dfQuestion, dfUsers, dfInteractions)

    # Get recommendations
    collab = recommend_questions_collab(user_id, dfInteractions, num_questions * 2)
    content = recommend_questions_content(user_id, dfUsers, dfQuestion, dfInteractions, num_questions * 2)
    bandit_df = recommender.recommend(user_id, top_n=num_questions * 2)
    bandit = bandit_df['question_id'].tolist()
    job_title_based = recommend_questions_job_title_only(dfQuestion, dfInteractions, user_id, num_questions * 3)

     # TEST
    print("Collaborative Filtering (CF):", collab)
    print("Content-Based Filtering (CBF):", content)
    print("Bandit Recommendations:", bandit)
    print("Job Title Matched:", job_title_based)

    score_dict = {}
    for i, qid in enumerate(collab):
        score_dict[qid] = score_dict.get(qid, 0) + alpha * (1 / (i + 1))
    for i, qid in enumerate(content):
        score_dict[qid] = score_dict.get(qid, 0) + beta * (1 / (i + 1))
    for i, qid in enumerate(bandit):
        score_dict[qid] = score_dict.get(qid, 0) + gamma * (1 / (i + 1))
    for qid in score_dict:
        if qid in job_title_based:
            score_dict[qid] += delta

    answered = get_answered_questions(dfInteractions, user_id)
    ranked = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    top_qids = [qid for qid, _ in ranked if qid not in answered][:num_questions]

    print("Final Hybrid Recommendation:", top_qids)
    return dfQuestion[dfQuestion['question_id'].isin(top_qids)]


