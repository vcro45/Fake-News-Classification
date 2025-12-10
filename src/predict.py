# predict.py
# Prediction pipeline for fake news detection
# This script loads the saved model and vectorizer, and provides prediction functions
# It will serve as the core of the website backend

import pickle
import os

# ---------------------------
# CONFIGURATION
# ---------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "logistic_model.pkl")

# Credibility thresholds
THRESHOLD_REAL = 0.75      # p >= 0.75 → Likely Real
THRESHOLD_UNCERTAIN = 0.40  # 0.40 <= p < 0.75 → Uncertain
                            # p < 0.40 → Likely Fake

# ---------------------------
# LOAD MODEL AND VECTORIZER
# ---------------------------
def load_model():
    """Load the saved TF-IDF vectorizer and Logistic Regression model."""
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return vectorizer, model

# Global model loading (load once when module is imported)
_vectorizer, _model = load_model()

# ---------------------------
# PREDICTION FUNCTIONS
# ---------------------------
def get_credibility_score(text):
    """
    Get the credibility score (probability of being REAL) for a given text.
    
    Args:
        text (str): The news article or headline to analyze
        
    Returns:
        float: Probability between 0 and 1 (higher = more likely real)
    """
    text_vec = _vectorizer.transform([text])
    prob_real = _model.predict_proba(text_vec)[0, 1]
    return prob_real

def get_credibility_label(score):
    """
    Convert credibility score to human-readable label.
    
    Args:
        score (float): Credibility score between 0 and 1
        
    Returns:
        str: "Likely Real", "Uncertain", or "Likely Fake"
    """
    if score >= THRESHOLD_REAL:
        return "Likely Real"
    elif score >= THRESHOLD_UNCERTAIN:
        return "Uncertain"
    else:
        return "Likely Fake"

def predict(text):
    """
    Full prediction pipeline for WEBSITE: returns score, label, and confidence.
    NO explainability features - clean output only.
    
    Args:
        text (str): The news article or headline to analyze
        
    Returns:
        dict: {
            "score": float (0-1),
            "label": str,
            "confidence": str
        }
    """
    score = get_credibility_score(text)
    label = get_credibility_label(score)
    
    # Confidence based on how far from decision boundaries
    if score >= 0.90 or score <= 0.10:
        confidence = "High"
    elif score >= 0.75 or score <= 0.25:
        confidence = "Medium"
    else:
        confidence = "Low"
    
    return {
        "score": round(score, 4),
        "label": label,
        "confidence": confidence
    }

# ---------------------------
# MAIN (for testing)
# ---------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("FAKE NEWS DETECTION - Prediction Pipeline")
    print("=" * 60)
    
    # Test examples
    test_texts = [
        "BREAKING: Scientists discover New York City is actually built on ancient alien ruins",
        "The Federal Reserve announced a 0.25% interest rate increase today, citing inflation concerns. The decision was made after careful review of economic indicators.",
        "You won't BELIEVE what this celebrity did! Doctors HATE this one weird trick!",
        "President signs bipartisan infrastructure bill into law after months of negotiations in Congress.",
    ]
    
    print("\nTesting prediction pipeline:\n")
    for i, text in enumerate(test_texts, 1):
        result = predict(text)
        print(f"[{i}] Text: {result['text']}")
        print(f"    Score: {result['score']:.4f}")
        print(f"    Label: {result['label']}")
        print(f"    Confidence: {result['confidence']}")
        print()
    
    print("=" * 60)
    print("Pipeline ready for backend integration!")
    print("=" * 60)
