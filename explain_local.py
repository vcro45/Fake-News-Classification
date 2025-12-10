# explain_local.py
# A2 — Local Explainability: Per-text word importance
# For any input text, get word-level importance scores

import numpy as np
import joblib
import json

# ---------------------------
# LOAD MODEL AND VECTORIZER
# ---------------------------
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
model = joblib.load("model/logistic_model.pkl")

feature_names = np.array(vectorizer.get_feature_names_out())
coef = model.coef_[0]  # Logistic regression coefficients

# ---------------------------
# LOCAL EXPLANATION FUNCTION
# ---------------------------
def explain_text(text, top_k=15):
    """
    Get word-level importance scores for a given text.
    
    Args:
        text (str): The news article or headline to analyze
        top_k (int): Number of top important words to return
        
    Returns:
        dict: {
            "text": original text,
            "prediction_score": float (0-1, probability of REAL),
            "prediction_label": str,
            "word_importance": [{"token": str, "weight": float}, ...]
        }
    """
    # 1) Get TF-IDF vector for the text
    tfidf_vec = vectorizer.transform([text])
    
    # 2) Get prediction probability
    prob_real = model.predict_proba(tfidf_vec)[0, 1]
    
    # 3) Get label
    if prob_real >= 0.75:
        label = "Likely Real"
    elif prob_real >= 0.40:
        label = "Uncertain"
    else:
        label = "Likely Fake"
    
    # 4) Calculate word importance: TF-IDF weight × coefficient
    tfidf_array = tfidf_vec.toarray()[0]
    word_importance = tfidf_array * coef
    
    # 5) Get non-zero indices (words that appear in the text)
    nonzero_idx = np.where(tfidf_array > 0)[0]
    
    # 6) Build list of (word, importance) for words in the text
    importance_list = []
    for idx in nonzero_idx:
        importance_list.append({
            "token": feature_names[idx],
            "weight": round(float(word_importance[idx]), 4),
            "tfidf": round(float(tfidf_array[idx]), 4)
        })
    
    # 7) Sort by absolute importance (most influential first)
    importance_list.sort(key=lambda x: abs(x["weight"]), reverse=True)
    
    # 8) Take top_k
    top_words = importance_list[:top_k]
    
    return {
        "text": text[:300] + "..." if len(text) > 300 else text,
        "prediction_score": round(float(prob_real), 4),
        "prediction_label": label,
        "word_importance": top_words
    }


def explain_text_simple(text, top_k=10):
    """
    Simplified version returning just token and weight.
    Suitable for frontend highlighting.
    """
    result = explain_text(text, top_k)
    return {
        "score": result["prediction_score"],
        "label": result["prediction_label"],
        "word_importance": [
            {"token": w["token"], "weight": w["weight"]} 
            for w in result["word_importance"]
        ]
    }


# ---------------------------
# MAIN (for testing)
# ---------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("A2 — Local Explainability: Per-Text Word Importance")
    print("=" * 60)
    
    # Test examples
    test_texts = [
        "BREAKING: Scientists discover New York City is actually built on ancient alien ruins. Watch the shocking video!",
        "WASHINGTON (Reuters) - The Federal Reserve announced a 0.25% interest rate increase on Wednesday, citing inflation concerns. The decision was made after careful review of economic indicators, said a spokesman.",
        "You won't BELIEVE what Hillary Clinton did! Obama EXPOSED in shocking scandal. Share this before it gets deleted!",
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n[Example {i}]")
        print(f"Text: {text[:100]}...")
        
        result = explain_text(text)
        
        print(f"\nPrediction: {result['prediction_label']} (score: {result['prediction_score']:.4f})")
        print("\nTop influential words:")
        print("-" * 40)
        
        for w in result["word_importance"]:
            direction = "→ REAL" if w["weight"] > 0 else "→ FAKE"
            print(f"  {w['token']:20s}  {w['weight']:+.4f}  {direction}")
        
        print()
    
    # Show JSON output format for frontend
    print("=" * 60)
    print("JSON Output Format (for frontend):")
    print("=" * 60)
    simple_result = explain_text_simple(test_texts[0])
    print(json.dumps(simple_result, indent=2))
