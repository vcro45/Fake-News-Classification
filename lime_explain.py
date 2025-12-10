# lime_explain.py
# A3 — LIME explainability for Fake News Classifier (TF-IDF + Logistic Regression)

import os
import numpy as np
from lime.lime_text import LimeTextExplainer
import pickle

# Load vectorizer + model (using existing paths)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "logistic_model.pkl")

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

CLASS_NAMES = ["Fake", "Real"]  # LR model outputs 0=fake, 1=real

# Ensure directory exists
os.makedirs("results/explanations", exist_ok=True)


def explain_with_lime(text, num_features=10, save_html=True, filename="lime_explanation.html"):
    """
    Generates a LIME explanation for a given news article.
    
    Args:
        text (str): The news article or headline to analyze
        num_features (int): Number of top features to show in explanation
        save_html (bool): Whether to save HTML visualization
        filename (str): Name of the HTML file to save
    
    Returns:
        dict: {
            "text": original text,
            "score": probability_real,
            "label": "Likely Fake" / "Uncertain" / "Likely Real",
            "confidence": "Low/Medium/High",
            "explanation_json": [ {token, weight}, ... ],
            "html_path": ".../lime_explanation.html"
        }
    """

    explainer = LimeTextExplainer(class_names=CLASS_NAMES)

    # Wrapper for LIME → uses vectorizer + model.predict_proba
    def predict_fn(texts):
        X = vectorizer.transform(texts)
        return model.predict_proba(X)

    # Generate explanation
    exp = explainer.explain_instance(
        text_instance=text,
        classifier_fn=predict_fn,
        num_features=num_features
    )

    # Convert to JSON format
    weights = exp.as_list()
    explanation_json = [
        {"token": w[0], "weight": round(float(w[1]), 4)} 
        for w in weights
    ]

    # Save HTML file
    html_path = None
    if save_html:
        html_path = f"results/explanations/{filename}"
        exp.save_to_file(html_path)

    # Final prediction
    proba_real = model.predict_proba(vectorizer.transform([text]))[0][1]

    # Labeling logic (same thresholds as main pipeline)
    if proba_real >= 0.75:
        label = "Likely Real"
        confidence = "High" if proba_real >= 0.90 else "Medium"
    elif proba_real >= 0.40:
        label = "Uncertain"
        confidence = "Low"
    else:
        label = "Likely Fake"
        confidence = "High" if proba_real <= 0.10 else "Medium"

    return {
        "text": text[:300] + "..." if len(text) > 300 else text,
        "score": round(float(proba_real), 4),
        "label": label,
        "confidence": confidence,
        "explanation_json": explanation_json,
        "html_path": html_path
    }


# ---------------------------
# MAIN (for testing)
# ---------------------------
if __name__ == "__main__":
    import json
    
    print("=" * 60)
    print("A3 — LIME Explainability Test")
    print("=" * 60)
    
    test_texts = [
        "BREAKING: Hillary Clinton involved in shocking scandal, video reveals truth!",
        "WASHINGTON (Reuters) - The Senate passed a bipartisan infrastructure bill on Thursday after months of negotiations, said a spokesman.",
        "You won't BELIEVE what Obama did! Watch this video before it gets deleted!"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n[Example {i}]")
        print(f"Text: {text[:80]}...")
        
        result = explain_with_lime(text, num_features=8, filename=f"lime_example_{i}.html")
        
        print(f"\nPrediction: {result['label']} (score: {result['score']:.4f}, confidence: {result['confidence']})")
        print("\nLIME Explanation (top words):")
        print("-" * 40)
        
        for item in result["explanation_json"]:
            direction = "→ REAL" if item["weight"] > 0 else "→ FAKE"
            print(f"  {item['token']:20s}  {item['weight']:+.4f}  {direction}")
        
        print(f"\nHTML saved: {result['html_path']}")
    
    print("\n" + "=" * 60)
    print("JSON Output Format:")
    print("=" * 60)
    print(json.dumps(result, indent=2))
