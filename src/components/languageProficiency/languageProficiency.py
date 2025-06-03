import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# Custom Dataset class
class RegressionDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.targets = targets

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.targets[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.targets)

def train_language_model(csv_path: str, model_save_path: str):
    df = pd.read_csv(csv_path)[["full_text", "Overall"]].dropna().reset_index(drop=True)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)

    train_texts, val_texts, train_scores, val_scores = train_test_split(
        df["full_text"].tolist(),
        df["Overall"].tolist(),
        test_size=0.2,
        random_state=42
    )

    train_dataset = RegressionDataset(train_texts, train_scores, tokenizer)
    val_dataset = RegressionDataset(val_texts, val_scores, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    model.train()
    for epoch in range(3):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")

    # Evaluate MSE
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = outputs.logits.detach().cpu().numpy().flatten()
            labels = batch["labels"].cpu().numpy().flatten()
            predictions.extend(preds)
            actuals.extend(labels)

    mse = mean_squared_error(actuals, predictions)
    print(f"\nValidation MSE: {mse:.4f}")

    # Save model and tokenizer
    os.makedirs(model_save_path, exist_ok=True)
    model.save_pretrained(os.path.join(model_save_path, "model"))
    tokenizer.save_pretrained(os.path.join(model_save_path, "tokenizer"))
    print(f"\n✅ Model saved to: {model_save_path}")


# This block makes sure the function runs only when this file is executed directly
if __name__ == "__main__":
    csv_path = "notebook/data/languageProficiency/listed.csv"
    model_save_path = "artifact/languageProficiency"
    train_language_model(csv_path, model_save_path)


