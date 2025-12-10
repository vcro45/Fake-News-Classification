# test_headline.py
# Interactive headline tester - Baseline Model Only (fast)
# For transformer model, use: python test_headline_transformer.py "headline"

import sys

# Get headline from command line argument
if len(sys.argv) < 2:
    print("Usage: python test_headline.py 'Your headline here'")
    sys.exit(1)

headline = sys.argv[1]

print("\n" + "=" * 60)
print(f"Testing: {headline}")
print("=" * 60)

# Baseline Model
from predict import predict
result = predict(headline)

print(f"\n📊 BASELINE MODEL (TF-IDF + Logistic Regression)")
print(f"   Score: {result['score']:.4f}")
print(f"   Label: {result['label']}")
print(f"   Confidence: {result['confidence']}")
print("\n" + "=" * 60)
