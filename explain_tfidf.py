# explain_tfidf.py
# A1 — Global Explainability: Top TF-IDF / Logistic Regression Words

import os
import numpy as np
import joblib
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

# 1) Load model + vectorizer
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
model = joblib.load("model/logistic_model.pkl")

feature_names = np.array(vectorizer.get_feature_names_out())
coef = model.coef_[0]  # binary classifier: single coefficient vector

# Positive coef → push toward REAL (label 1)
# Negative coef → push toward FAKE (label 0)

TOP_K = 20

# 2) Top REAL-leaning words
top_real_idx = np.argsort(coef)[-TOP_K:]
top_real_words = feature_names[top_real_idx]
top_real_values = coef[top_real_idx]

# 3) Top FAKE-leaning words
top_fake_idx = np.argsort(coef)[:TOP_K]
top_fake_words = feature_names[top_fake_idx]
top_fake_values = coef[top_fake_idx]

print("=== Top REAL-leaning words ===")
for w, v in zip(top_real_words, top_real_values):
    print(f"{w:20s}  {v:.3f}")

print("\n=== Top FAKE-leaning words ===")
for w, v in zip(top_fake_words, top_fake_values):
    print(f"{w:20s}  {v:.3f}")

# 4) Save bar plots

def plot_top_words(words, values, title, filename):
    order = np.argsort(values)
    words = words[order]
    values = values[order]

    plt.figure(figsize=(8, 6))
    plt.barh(range(len(words)), values)
    plt.yticks(range(len(words)), words)
    plt.xlabel("Logistic Regression Coefficient")
    plt.title(title)
    plt.tight_layout()
    out_path = os.path.join("results", filename)
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved plot: {out_path}")

plot_top_words(top_real_words, top_real_values,
               "Top REAL-Leaning Words", "top_real_words.png")

plot_top_words(top_fake_words, top_fake_values,
               "Top FAKE-Leaning Words", "top_fake_words.png")
