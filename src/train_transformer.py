# train_transformer.py
# B4 — Advanced Model: DistilBERT embeddings + Logistic Regression
# This script:
# 1. Loads merged_clean.csv
# 2. Generates DistilBERT embeddings (768-dim) for each article
# 3. Trains Logistic Regression on top
# 4. Evaluates and compares with baseline
# 5. Saves model and plots

import pandas as pd
import numpy as np
import pickle
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    classification_report
)

import torch
from transformers import DistilBertTokenizer, DistilBertModel

# ---------------------------
# CONFIGURATION
# ---------------------------
DATA_PATH = "../data/merged_clean.csv"
MODEL_DIR = "../model"
RESULTS_DIR = "../results"
BATCH_SIZE = 32
MAX_LENGTH = 512  # DistilBERT max token length
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"[INFO] Using device: {DEVICE}")

# ---------------------------
# STEP 1: LOAD DATA
# ---------------------------
print("[INFO] Loading data...")
df = pd.read_csv(DATA_PATH)

# Use a subset for faster training (optional - remove for full training)
# Uncomment below line to use full dataset (will take longer)
# df = df  # Full dataset

# For faster initial training, use subset
SAMPLE_SIZE = 10000  # Adjust based on your time/compute
if len(df) > SAMPLE_SIZE:
    print(f"[INFO] Using subset of {SAMPLE_SIZE} samples for faster training...")
    df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

X = df["text"].tolist()
y = df["label"].values

print(f"[INFO] Dataset size: {len(X)} samples")

# ---------------------------
# STEP 2: TRAIN/VAL/TEST SPLIT
# ---------------------------
print("[INFO] Splitting data...")
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"  Train: {len(X_train)}")
print(f"  Val:   {len(X_val)}")
print(f"  Test:  {len(X_test)}")

# ---------------------------
# STEP 3: LOAD DISTILBERT
# ---------------------------
print("[INFO] Loading DistilBERT model...")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
bert_model.to(DEVICE)
bert_model.eval()

# ---------------------------
# STEP 4: GENERATE EMBEDDINGS
# ---------------------------
def get_embeddings(texts, batch_size=BATCH_SIZE):
    """Generate DistilBERT embeddings for a list of texts."""
    embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(DEVICE)
        attention_mask = encoded['attention_mask'].to(DEVICE)
        
        # Get embeddings
        with torch.no_grad():
            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            # Use [CLS] token embedding (first token)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embeddings)
    
    return np.vstack(embeddings)

print("\n[INFO] Generating embeddings for training set...")
X_train_emb = get_embeddings(X_train)

print("[INFO] Generating embeddings for validation set...")
X_val_emb = get_embeddings(X_val)

print("[INFO] Generating embeddings for test set...")
X_test_emb = get_embeddings(X_test)

print(f"\n[INFO] Embedding shape: {X_train_emb.shape}")

# ---------------------------
# STEP 5: TRAIN CLASSIFIER
# ---------------------------
print("\n[INFO] Training Logistic Regression on DistilBERT embeddings...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_emb, y_train)

# ---------------------------
# STEP 6: EVALUATION
# ---------------------------
print("\n[INFO] Evaluating model...")

# Validation predictions
y_val_probs = model.predict_proba(X_val_emb)[:, 1]
y_val_pred = (y_val_probs >= 0.5).astype(int)

# Metrics
acc = accuracy_score(y_val, y_val_pred)
prec = precision_score(y_val, y_val_pred)
rec = recall_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)

print("\n" + "=" * 50)
print("DISTILBERT + LOGISTIC REGRESSION - VALIDATION METRICS")
print("=" * 50)
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_val, y_val_pred, target_names=["Fake", "Real"]))

# ---------------------------
# STEP 7: CONFUSION MATRIX
# ---------------------------
cm = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Fake", "Real"],
            yticklabels=["Fake", "Real"])
plt.title("Confusion Matrix - DistilBERT Model")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_transformer.png"))
plt.close()
print(f"[INFO] Saved confusion matrix to {RESULTS_DIR}/confusion_matrix_transformer.png")

# ---------------------------
# STEP 8: ROC CURVE
# ---------------------------
fpr, tpr, _ = roc_curve(y_val, y_val_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"DistilBERT (AUC = {roc_auc:.3f})", linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - DistilBERT Model")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "roc_curve_transformer.png"))
plt.close()
print(f"[INFO] Saved ROC curve to {RESULTS_DIR}/roc_curve_transformer.png")

# ---------------------------
# STEP 9: PRECISION-RECALL CURVE
# ---------------------------
precision_curve, recall_curve, _ = precision_recall_curve(y_val, y_val_probs)
pr_auc = auc(recall_curve, precision_curve)

plt.figure(figsize=(8, 6))
plt.plot(recall_curve, precision_curve, label=f"DistilBERT (AUPR = {pr_auc:.3f})", linewidth=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - DistilBERT Model")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "pr_curve_transformer.png"))
plt.close()
print(f"[INFO] Saved PR curve to {RESULTS_DIR}/pr_curve_transformer.png")

# ---------------------------
# STEP 10: SAVE MODEL
# ---------------------------
print("\n[INFO] Saving model...")

# Save the classifier
with open(os.path.join(MODEL_DIR, "distilbert_classifier.pkl"), "wb") as f:
    pickle.dump(model, f)

# Save metrics for comparison
metrics = {
    "model": "DistilBERT + Logistic Regression",
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1_score": f1,
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
    "train_size": len(X_train),
    "val_size": len(X_val),
    "test_size": len(X_test)
}

with open(os.path.join(RESULTS_DIR, "transformer_metrics.pkl"), "wb") as f:
    pickle.dump(metrics, f)

print(f"[INFO] Saved classifier to {MODEL_DIR}/distilbert_classifier.pkl")
print(f"[INFO] Saved metrics to {RESULTS_DIR}/transformer_metrics.pkl")

# ---------------------------
# STEP 11: TEST SET EVALUATION
# ---------------------------
print("\n[INFO] Final evaluation on TEST set...")
y_test_probs = model.predict_proba(X_test_emb)[:, 1]
y_test_pred = (y_test_probs >= 0.5).astype(int)

test_acc = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

print("\n" + "=" * 50)
print("TEST SET METRICS")
print("=" * 50)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")

print("\n" + "=" * 50)
print("TRAINING COMPLETE!")
print("=" * 50)
