import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load model and tokenizer once
MODEL_DIR = "artifact/languageProficiency/model"
TOKENIZER_DIR = "artifact/languageProficiency/tokenizer"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

def predict_score(text: str) -> float:
    """
    Predict the language proficiency score for the given text.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        score = outputs.logits.item()

    return score
# Example usage
# if __name__ == "__main__":
#     sample_text = "The explanation in the answer was clear and fluent."
#     score = predict_score(sample_text)
#     print(f"Predicted score: {score:.2f}")