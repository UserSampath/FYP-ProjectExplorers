# load_and_recommend.py

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import re
from src.utils import get_engine


engine = get_engine()

dfQuestion = pd.read_sql("SELECT * FROM processed_question", engine)
dfUsers = pd.read_sql("SELECT * FROM processed_users", engine)
dfInteractions = pd.read_sql("SELECT * FROM processed_interactions", engine)
dfJobTitles = pd.read_sql("SELECT * FROM cleaned_job_titles ORDER BY id DESC LIMIT 500", engine)


# # Load preprocessed data
# dfQuestion = pd.read_csv("artifact/question_recommendation/processed_question.csv")
# dfUsers = pd.read_csv("artifact/question_recommendation/processed_users.csv")
# dfInteractions = pd.read_csv("artifact/question_recommendation/processed_interactions.csv")
# dfJobTitles = pd.read_csv("artifact/question_recommendation/cleaned_job_titles.csv")

# Collaborative Filtering (KNN + SVD)
dfInteractions['weighted_score'] = (
    dfInteractions['answered_correctly'] + dfInteractions['timeTaken_minmax'] + dfInteractions['difficulty_encoded']
) / 3

interaction_matrix = dfInteractions.pivot(index='user_id', columns='question_id', values='weighted_score').fillna(0)
interaction_np = interaction_matrix.values

knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(interaction_np)

U, sigma, Vt = svds(interaction_np, k=5)
sigma = np.diag(sigma)
predicted = np.dot(np.dot(U, sigma), Vt)
predicted_df = pd.DataFrame(predicted, index=interaction_matrix.index, columns=interaction_matrix.columns)

def get_answered_questions(user_id):
    return set(dfInteractions[dfInteractions['user_id'] == user_id]['question_id'].tolist())

def recommend_questions_collab(user_id, n=5):
    if user_id not in predicted_df.index:
        return []
    answered = get_answered_questions(user_id)
    ranked = predicted_df.loc[user_id].sort_values(ascending=False)
    return [qid for qid in ranked.index if qid not in answered][:n]

# Content-Based Filtering (TF-IDF + LSA)
dfQuestion.fillna('', inplace=True)
dfQuestion['combined'] = dfQuestion[['topic', 'tags', 'question']].agg(' '.join, axis=1)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(dfQuestion['combined'])
lsa = TruncatedSVD(n_components=50)
lsa_matrix = lsa.fit_transform(tfidf_matrix)

def recommend_questions_content(user_id, n=5):
    user = dfUsers[dfUsers['user_id'] == user_id]
    if user.empty:
        return []
    answered = get_answered_questions(user_id)
    prefs = ' '.join(user['familiar_technologies'].astype(str).tolist())
    user_vec = vectorizer.transform([prefs])
    scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
    top_idx = scores.argsort()[::-1]
    recs = [dfQuestion.iloc[i]['question_id'] for i in top_idx if dfQuestion.iloc[i]['question_id'] not in answered]
    return recs[:n]

# Reinforcement Learning (Bandit)
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
                self.successes[qid] += 1 if t <= 30 else 0.8 if t <= 60 else 0.5

    def ucb_score(self, qid, total):
        mean = self.successes[qid] / self.attempts[qid]
        return mean + np.sqrt((2 * np.log(total)) / self.attempts[qid])

    def recommend(self, user_id, top_n=5):
        user = self.udf[self.udf['user_id'] == user_id]
        if user.empty:
            return pd.DataFrame()
        user = user.iloc[0]
        user_techs = [t.strip().lower() for t in str(user['familiar_technologies']).split(',')]
        level = str(user['expertise_level']).lower()
        answered = set(self.idf[self.idf['user_id'] == user_id]['question_id'])
        candidates = self.qdf[~self.qdf['question_id'].isin(answered)]
        total_attempts = sum(self.attempts.values())
        scored = []
        for _, q in candidates.iterrows():
            qid = q['question_id']
            score = self.ucb_score(qid, total_attempts)
            tags = [t.strip().lower() for t in str(q['tags']).split(',')]
            topic_match = any(tech in tags for tech in user_techs)
            difficulty_match = q['difficulty_level'].strip().lower() == level
            if topic_match: score *= 1.2
            if difficulty_match: score *= 1.1
            scored.append((qid, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = [qid for qid, _ in scored[:top_n]]
        return self.qdf[self.qdf['question_id'].isin(top)]

recommender = QuestionBanditRecommender(dfQuestion, dfUsers, dfInteractions)

# Job Title Matching
unwanted = ["se", "software engineer", "associate", "full stack", "fullstack", "developer", "designer", "engineer"]
pattern = r'\b(?:' + '|'.join(map(re.escape, unwanted)) + r')\b'
dfJobTitles['title'] = dfJobTitles['title'].str.lower().str.replace(pattern, '', regex=True)
dfJobTitles['title'] = dfJobTitles['title'].str.replace(r'\s+', ' ', regex=True).str.strip()
job_keywords = dfJobTitles['title'][dfJobTitles['title'].str.strip() != ''].unique().tolist()

str_cols = dfQuestion.select_dtypes(include=['object']).columns
dfQuestion[str_cols] = dfQuestion[str_cols].apply(lambda col: col.str.lower())
def match_title(row):
    return any(any(k in str(row[col]) for k in job_keywords) for col in ['question', 'topic', 'tags'])
dfQuestion['job_title_match'] = dfQuestion.apply(match_title, axis=1).astype(int)

def recommend_questions_job_title_only(user_id, n=10):
    answered = get_answered_questions(user_id)
    job_related = dfQuestion[(dfQuestion['job_title_match'] == 1) & (~dfQuestion['question_id'].isin(answered))]
    return job_related.head(n)['question_id'].tolist()

# Hybrid Recommendation
def hybrid_recommendations(user_id, num_questions=5, alpha=0.35, beta=0.25, gamma=0.25, delta=0.15):
    collab = recommend_questions_collab(user_id, num_questions * 2)
    content = recommend_questions_content(user_id, num_questions * 2)
    bandit = recommender.recommend(user_id, top_n=num_questions * 2)['question_id'].tolist()
    score_dict = {}
    for i, qid in enumerate(collab): score_dict[qid] = score_dict.get(qid, 0) + alpha * (1 / (i + 1))
    for i, qid in enumerate(content): score_dict[qid] = score_dict.get(qid, 0) + beta * (1 / (i + 1))
    for i, qid in enumerate(bandit): score_dict[qid] = score_dict.get(qid, 0) + gamma * (1 / (i + 1))
    for qid in score_dict:
        if dfQuestion[dfQuestion['question_id'] == qid]['job_title_match'].values[0]:
            score_dict[qid] += delta
    ranked = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    top_qids = [qid for qid, _ in ranked[:num_questions]]
    return dfQuestion[dfQuestion['question_id'].isin(top_qids)]

# Test
# user_id = 2
# print("CF:", recommend_questions_collab(user_id))
# print("CBF:", recommend_questions_content(user_id))
# print("Bandit:", recommender.recommend(user_id)['question_id'].tolist())
# print("Job Title:", recommend_questions_job_title_only(user_id))
# print("Hybrid:", hybrid_recommendations(user_id)['question_id'].tolist())
