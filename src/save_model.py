# save_model.py
# Save the trained TF-IDF vectorizer and Logistic Regression model

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ---------------------------
# STEP 1: LOAD MERGED DATASET
# ---------------------------
print("[INFO] Loading merged_clean.csv ...")
df = pd.read_csv("../data/merged_clean.csv")

X = df["text"]
y = df["label"]  # 0 = Fake, 1 = Real

# ------------------------------------
# STEP 2: TRAIN/VAL/TEST SPLIT
# ------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"[INFO] Training on {len(X_train)} samples...")

# ------------------------------------
# STEP 3: TF-IDF VECTORIZATION
# ------------------------------------
print("[INFO] Fitting TF-IDF vectorizer ...")
vectorizer = TfidfVectorizer(
    max_features=50000,
    stop_words='english',
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)

# ------------------------------------
# STEP 4: TRAIN LOGISTIC REGRESSION
# ------------------------------------
print("[INFO] Training Logistic Regression model ...")
model = LogisticRegression(max_iter=3000)
model.fit(X_train_vec, y_train)

# ------------------------------------
# STEP 5: SAVE MODEL AND VECTORIZER
# ------------------------------------
print("[INFO] Saving model and vectorizer ...")

with open("../model/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("../model/logistic_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("[INFO] Saved:")
print("  - model/tfidf_vectorizer.pkl")
print("  - model/logistic_model.pkl")
print("[INFO] Done!")
