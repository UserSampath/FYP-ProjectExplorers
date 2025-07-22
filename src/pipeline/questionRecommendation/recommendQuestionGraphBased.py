import pandas as pd
import numpy as np
from neo4j import GraphDatabase

from src.utils import get_engine

engine = get_engine()
dfQuestion = pd.read_sql("SELECT * FROM processed_question", engine)
dfUsers = pd.read_sql("SELECT * FROM processed_users", engine)
dfInteractions = pd.read_sql("SELECT * FROM processed_interactions", engine)

# Neo4j connection parameters
uri = "neo4j+s://46f0ed2d.databases.neo4j.io"
username = "neo4j"
password = "ZMivcvxHF7XHYbD1T_hkMsMBnBxNof30ThSJOnAfkkA"

driver = GraphDatabase.driver(uri, auth=(username, password))

def clear_graph(driver):
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("Graph cleared: all nodes and relationships deleted.")


updated_user = {
    "user_id": "997", 
    "expertise_level": "Intermediate",
    "years_of_experience": 3,
    "familiar_technologies": "Python,Machine Learning"
}

def update_existing_user(user):
    print(user,"updating user")
    with driver.session() as session:
        session.execute_write(create_user, user)
        if 'familiar_technologies' in user and isinstance(user['familiar_technologies'], str):
            session.execute_write(create_user_question_tech_relation, user)

new_user = {
    "user_id": "997",

}
def add_new_user(user):
    print(user,"user added")
    with driver.session() as session:
        session.execute_write(create_user, user)
        if 'familiar_technologies' in user and user['familiar_technologies'].strip():
            session.execute_write(create_user_question_tech_relation, user)


new_interaction = {
    "user_id": "999",                
    "question_id": "Q123",         
    "answered_correctly": True,      
    "created_at": "2025-07-20"       
}
def add_new_interaction(new_interaction):
    with driver.session() as session:
        session.execute_write(create_interaction, new_interaction)

def create_user(tx, user):
    # Always ensure the user node exists
    tx.run("""
        MERGE (u:User {user_id: $user_id})
    """, user_id=user['user_id'])

    # Conditionally update properties if present
    if 'expertise_level' in user:
        tx.run("""
            MATCH (u:User {user_id: $user_id})
            SET u.expertise_level = $expertise_level
        """, user_id=user['user_id'], expertise_level=user['expertise_level'])

    if 'years_of_experience' in user:
        tx.run("""
            MATCH (u:User {user_id: $user_id})
            SET u.years_of_experience = $years_of_experience
        """, user_id=user['user_id'], years_of_experience=user['years_of_experience'])


    if 'familiar_technologies' in user and isinstance(user['familiar_technologies'], str):
        techs = [t.strip() for t in user['familiar_technologies'].split(",")]

        for tech in techs:
            tx.run("""
                MERGE (t:Technology {name: $tech})
                WITH t
                MATCH (u:User {user_id: $user_id})
                MERGE (u)-[:FAMILIAR_WITH]->(t)
            """, tech=tech, user_id=user['user_id'])

        for tag in techs:
            tx.run("""
                MERGE (tag:Tag {name: $tag})
                WITH tag
                MATCH (u:User {user_id: $user_id})
                MERGE (u)-[:INTERESTED_IN]->(tag)
            """, tag=tag, user_id=user['user_id'])



def create_question(tx, question):
    tx.run("""
        MERGE (q:Question {question_id: $question_id})
        SET q.question = $question,
            q.option_A = $option_A,
            q.option_B = $option_B,
            q.option_C = $option_C,
            q.option_D = $option_D,
            q.correct_option = $correct_option,
            q.difficulty_level = $difficulty_level,
            q.category = $category,
            q.topic = $topic,
            q.tags = $tags
    """, question_id=question['question_id'],
         question=question['question'],
         option_A=question['option_A'],
         option_B=question['option_B'],
         option_C=question['option_C'],
         option_D=question['option_D'],
         correct_option=question['correct_option'],
         difficulty_level=question['difficulty_level'],
         category=question['category'],
         topic=question['topic'],
         tags=question['tags'])

    # Link question to Technology nodes based on topic and/or category
    tech_candidates = []
    if isinstance(question['topic'], str) and question['topic'].strip():
        tech_candidates.append(question['topic'].strip())
    if isinstance(question['category'], str) and question['category'].strip():
        tech_candidates.append(question['category'].strip())
    tech_candidates = list(set(tech_candidates))  # unique

    for tech in tech_candidates:
        tx.run("""
            MERGE (t:Technology {name: $tech})
            WITH t
            MATCH (q:Question {question_id: $question_id})
            MERGE (q)-[:RELATED_TO]->(t)
        """, tech=tech, question_id=question['question_id'])

    # Create Tag nodes and HAS_TAG relationships
    tags = question['tags']
    if isinstance(tags, str) and tags.strip():
        tag_list = [tag.strip() for tag in tags.split(",")]
        for tag in tag_list:
            tx.run("""
                MERGE (tag:Tag {name: $tag})
                WITH tag
                MATCH (q:Question {question_id: $question_id})
                MERGE (q)-[:HAS_TAG]->(tag)
            """, tag=tag, question_id=question['question_id'])


def create_interaction(tx, interaction):
    tx.run("""
        MATCH (u:User {user_id: $user_id}), (q:Question {question_id: $question_id})
        MERGE (u)-[r:ANSWERED]->(q)
        SET r.answered_correctly = $answered_correctly,
        r.created_at = $created_at
    """, user_id=interaction['user_id'],
         question_id=interaction['question_id'],
         answered_correctly=interaction['answered_correctly'],
         created_at=str(interaction.get('created_at', '')))


def create_user_question_tech_relation(tx, user):
    # Create direct relationship between user and questions that relate to user's familiar technologies
    if isinstance(user['familiar_technologies'], str):
        techs = [t.strip() for t in user['familiar_technologies'].split(",")]
        for tech in techs:
            tx.run("""
                MATCH (u:User {user_id: $user_id}), (q:Question)-[:RELATED_TO]->(t:Technology {name: $tech})
                MERGE (u)-[:FAMILIAR_WITH_QUESTION]->(q)
            """, user_id=user['user_id'], tech=tech)


def build_graph(dfUsers, dfQuestion, dfInteractions):
    with driver.session() as session:
        print("Creating User nodes and FAMILIAR_WITH relationships...")
        for _, user in dfUsers.iterrows():
            session.write_transaction(create_user, user)

        print("Creating Question nodes, RELATED_TO and HAS_TAG relationships...")
        for _, question in dfQuestion.iterrows():
            session.write_transaction(create_question, question)

        print("Creating User-Question direct relationships based on familiar technologies...")
        for _, user in dfUsers.iterrows():
            session.write_transaction(create_user_question_tech_relation, user)

        print("Creating Interaction edges...")
        for _, interaction in dfInteractions.iterrows():
            session.write_transaction(create_interaction, interaction)

    print("Graph building complete!")


def recommend_questions_by_user(user_id):
    with driver.session() as session:
        result = session.run("""
            MATCH (u:User {user_id: $user_id})

            OPTIONAL MATCH (u)-[:FAMILIAR_WITH]->(t:Technology)<-[:RELATED_TO]-(q1:Question)
            OPTIONAL MATCH (u)-[:INTERESTED_IN]->(tag:Tag)<-[:HAS_TAG]-(q2:Question)
            OPTIONAL MATCH (u)-[:FAMILIAR_WITH_QUESTION]->(q3:Question)

            WITH u, 
                COLLECT(DISTINCT q1) + COLLECT(DISTINCT q2) + COLLECT(DISTINCT q3) AS candidate_questions
            UNWIND candidate_questions AS q
            WITH u, q, q.difficulty_level AS difficulty_level
            WHERE NOT (u)-[:ANSWERED]->(q)

            RETURN DISTINCT q.question_id AS question_id, difficulty_level
            ORDER BY difficulty_level
            LIMIT 40
        """, user_id=user_id)

        return [record["question_id"] for record in result]




if __name__ == "__main__":
    # Uncomment below lines to clear and build graph
    print("test")
    # clear_graph(driver)
    # build_graph(dfUsers, dfQuestion, dfInteractions)
    
    # result = recommend_questions_by_user_profile("998")
    # question_ids = [q['q.question_id'] for q in result]
    # print("Recommended Question IDs:", question_ids)

    # add_new_user(new_user)
    # add_new_interaction()
    # update_existing_user(updated_user)
