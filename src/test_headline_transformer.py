# test_headline_transformer.py
# Interactive headline tester - Transformer Model (slower but more accurate)

import sys

# Get headline from command line argument
if len(sys.argv) < 2:
    print("Usage: python test_headline_transformer.py 'Your headline here'")
    sys.exit(1)

headline = sys.argv[1]

print("\n" + "=" * 60)
print(f"Testing: {headline}")
print("=" * 60)

# Transformer Model
from predict_transformer import predict_transformer
result = predict_transformer(headline)

print(f"\n🤖 TRANSFORMER MODEL (DistilBERT + Logistic Regression)")
print(f"   Score: {result['score']:.4f}")
print(f"   Label: {result['label']}")
print(f"   Confidence: {result['confidence']}")
print("\n" + "=" * 60)
