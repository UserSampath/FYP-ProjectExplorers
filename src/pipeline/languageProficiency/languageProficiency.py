import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Model and tokenizer paths
COHESION_MODEL_DIR = "artifact/languageProficiency/cohesionModel"
COHESION_TOKENIZER_DIR = "artifact/languageProficiency/cohesionTokenizer"

GRAMMAR_MODEL_DIR = "artifact/languageProficiency/grammarModel"
GRAMMAR_TOKENIZER_DIR = "artifact/languageProficiency/grammarTokenizer"

SYNTAX_MODEL_DIR = "artifact/languageProficiency/syntaxModel"
SYNTAX_TOKENIZER_DIR = "artifact/languageProficiency/syntaxTokenizer"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load all models and tokenizers
cohesion_tokenizer = BertTokenizer.from_pretrained(COHESION_TOKENIZER_DIR)
cohesion_model = BertForSequenceClassification.from_pretrained(COHESION_MODEL_DIR).to(device).eval()

grammar_tokenizer = BertTokenizer.from_pretrained(GRAMMAR_TOKENIZER_DIR)
grammar_model = BertForSequenceClassification.from_pretrained(GRAMMAR_MODEL_DIR).to(device).eval()

syntax_tokenizer = BertTokenizer.from_pretrained(SYNTAX_TOKENIZER_DIR)
syntax_model = BertForSequenceClassification.from_pretrained(SYNTAX_MODEL_DIR).to(device).eval()


def predict_single_score(text: str, tokenizer, model) -> float:
    """
    Predict the score for a single aspect (cohesion/grammar/syntax).
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        score = outputs.logits.item()

    return score


def predict_all_scores(text: str) -> dict:
    cohesion_score = predict_single_score(text, cohesion_tokenizer, cohesion_model)
    grammar_score = predict_single_score(text, grammar_tokenizer, grammar_model)
    syntax_score = predict_single_score(text, syntax_tokenizer, syntax_model)

    average_score = (cohesion_score + grammar_score + syntax_score) / 3.0

    return {
        "cohesion": round(cohesion_score*2, 2),
        "grammar": round(grammar_score*2, 2),
        "syntax": round(syntax_score*2, 2),
        "overall": round(average_score*2, 2)
    }
