import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)

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

print(f"[INFO] Data split sizes:")
print(f" Train: {len(X_train)}")
print(f" Val:   {len(X_val)}")
print(f" Test:  {len(X_test)}")

# ------------------------------------
# STEP 3: TF-IDF VECTORIZATION
# ------------------------------------
print("[INFO] Vectorizing text using TF-IDF ...")
vectorizer = TfidfVectorizer(
    max_features=50000,
    stop_words='english',
    ngram_range=(1,2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)

# ------------------------------------
# STEP 4: TRAIN BASELINE MODEL
# ------------------------------------
print("[INFO] Training Logistic Regression baseline...")
model = LogisticRegression(max_iter=3000)
model.fit(X_train_vec, y_train)

# ------------------------------------
# STEP 5: EVALUATION
# ------------------------------------
y_val_probs = model.predict_proba(X_val_vec)[:, 1]   # probability of REAL
y_val_pred = (y_val_probs >= 0.5).astype(int)

acc = accuracy_score(y_val, y_val_pred)
prec = precision_score(y_val, y_val_pred)
rec = recall_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)

print("\n===== VALIDATION METRICS =====")
print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1 Score:", f1)

# ------------------------------------
# STEP 6: CONFUSION MATRIX PLOT
# ------------------------------------
cm = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Fake", "Real"],
            yticklabels=["Fake", "Real"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("../results/confusion_matrix.png")
plt.close()

# ------------------------------------
# STEP 7: ROC CURVE
# ------------------------------------
fpr, tpr, _ = roc_curve(y_val, y_val_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("../results/roc_curve.png")
plt.close()

# ------------------------------------
# STEP 8: PRECISION-RECALL CURVE
# ------------------------------------
precision, recall, _ = precision_recall_curve(y_val, y_val_probs)
pr_auc = auc(recall, precision)

plt.figure()
plt.plot(recall, precision, label=f"AUPR = {pr_auc:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.savefig("../results/pr_curve.png")
plt.close()

print("\nSaved all evaluation plots in results/ folder")

# ------------------------------------
# STEP 9: MAP PROBABILITIES TO CATEGORIES (Option A)
# ------------------------------------
def credibility_category(p):
    if p >= 0.75:
        return "Likely Real"
    elif p >= 0.40:
        return "Uncertain"
    else:
        return "Likely Fake"

sample_text = X_test.iloc[0]
sample_vec = vectorizer.transform([sample_text])
sample_prob = model.predict_proba(sample_vec)[0,1]
print("\nExample credibility score:")
print("Text:", sample_text[:200], "...")
print("Probability REAL:", sample_prob)
print("Category:", credibility_category(sample_prob))
