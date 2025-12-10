# predict_transformer.py
# Prediction pipeline using DistilBERT embeddings + Logistic Regression classifier

import torch
import pickle
import os
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel

# ---------------------------
# CONFIGURATION
# ---------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "distilbert_classifier.pkl")

# Device selection: MPS (Apple Silicon) > CUDA > CPU
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Credibility thresholds
THRESHOLD_REAL = 0.75
THRESHOLD_UNCERTAIN = 0.40

# ---------------------------
# LOAD MODELS (once at import)
# ---------------------------
print("[INFO] Loading DistilBERT tokenizer and model...")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
bert_model.to(DEVICE)
bert_model.eval()

print("[INFO] Loading classifier...")
with open(CLASSIFIER_PATH, "rb") as f:
    classifier = pickle.load(f)

print(f"[INFO] Models loaded successfully! Using device: {DEVICE}")

# ---------------------------
# EMBEDDING FUNCTION
# ---------------------------
def embed_text(text: str) -> np.ndarray:
    """
    Generate DistilBERT embedding (768-dim) for a single text input.
    Uses [CLS] token representation.
    """
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)
    
    with torch.no_grad():
        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token embedding (first token)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    
    return embedding

# ---------------------------
# PREDICTION FUNCTIONS
# ---------------------------
def get_credibility_score(text: str) -> float:
    """Get probability of being REAL (0-1)."""
    embedding = embed_text(text)
    prob_real = classifier.predict_proba(embedding)[0][1]
    return float(prob_real)

def get_label(score: float) -> str:
    """Convert score to label."""
    if score >= THRESHOLD_REAL:
        return "Likely Real"
    elif score >= THRESHOLD_UNCERTAIN:
        return "Uncertain"
    else:
        return "Likely Fake"

def get_confidence(score: float) -> str:
    """Get confidence level based on score."""
    if score >= 0.90 or score <= 0.10:
        return "High"
    elif score >= 0.75 or score <= 0.25:
        return "Medium"
    else:
        return "Low"

def predict_transformer(text: str) -> dict:
    """
    Full prediction pipeline using DistilBERT + Logistic Regression.
    
    Args:
        text (str): News headline or article to analyze
        
    Returns:
        dict: {
            "score": float (0-1),
            "label": str,
            "confidence": str
        }
    """
    score = get_credibility_score(text)
    label = get_label(score)
    confidence = get_confidence(score)
    
    return {
        "score": round(score, 4),
        "label": label,
        "confidence": confidence
    }

# ---------------------------
# MAIN (for testing)
# ---------------------------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TRANSFORMER MODEL - Prediction Test")
    print("=" * 60)
    
    test_texts = [
        "NASA announces discovery of new habitable exoplanet.",
        "BREAKING: Hillary Clinton involved in shocking scandal!",
        "WASHINGTON (Reuters) - The Senate passed a bipartisan infrastructure bill on Thursday.",
        "You won't BELIEVE what Obama did! Watch this video before it gets deleted!",
        "U.S. Senate Passes Bipartisan Bill to Improve National Infrastructure",
    ]
    
    print("\nTesting predictions:\n")
    for i, text in enumerate(test_texts, 1):
        result = predict_transformer(text)
        print(f"[{i}] Text: {text[:70]}...")
        print(f"    Score: {result['score']:.4f}")
        print(f"    Label: {result['label']}")
        print(f"    Confidence: {result['confidence']}")
        print()
    
    print("=" * 60)
    print("Transformer prediction pipeline ready!")
    print("=" * 60)
