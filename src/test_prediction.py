# test_prediction.py
# Test runner for both baseline and transformer prediction models

print("=" * 60)
print("FAKE NEWS DETECTION - Model Testing")
print("=" * 60)

# Test headlines
test_headlines = [
    "BREAKING: Hillary Clinton involved in shocking scandal!",
    "WASHINGTON (Reuters) - The Senate passed a bipartisan infrastructure bill on Thursday.",
    "You won't BELIEVE what Obama did! Watch this video before it gets deleted!",
    "U.S. Senate Passes Bipartisan Bill to Improve National Infrastructure",
    "Trump gets shot in his rally.",
]

# ---------------------------
# TEST BASELINE MODEL (TF-IDF + Logistic Regression)
# ---------------------------
print("\n" + "=" * 60)
print("BASELINE MODEL (TF-IDF + Logistic Regression)")
print("=" * 60)

from predict import predict as predict_baseline

for i, headline in enumerate(test_headlines, 1):
    result = predict_baseline(headline)
    print(f"\n[{i}] {headline[:60]}...")
    print(f"    Score: {result['score']:.4f} | Label: {result['label']} | Confidence: {result['confidence']}")

# ---------------------------
# TEST TRANSFORMER MODEL (DistilBERT + Logistic Regression)
# ---------------------------
print("\n" + "=" * 60)
print("TRANSFORMER MODEL (DistilBERT + Logistic Regression)")
print("=" * 60)

from predict_transformer import predict_transformer

for i, headline in enumerate(test_headlines, 1):
    result = predict_transformer(headline)
    print(f"\n[{i}] {headline[:60]}...")
    print(f"    Score: {result['score']:.4f} | Label: {result['label']} | Confidence: {result['confidence']}")

# ---------------------------
# SIDE-BY-SIDE COMPARISON
# ---------------------------
print("\n" + "=" * 60)
print("SIDE-BY-SIDE COMPARISON")
print("=" * 60)
print(f"{'Headline':<50} | {'Baseline':^12} | {'Transformer':^12}")
print("-" * 80)

for headline in test_headlines:
    baseline_result = predict_baseline(headline)
    transformer_result = predict_transformer(headline)
    
    short_headline = headline[:47] + "..." if len(headline) > 50 else headline
    print(f"{short_headline:<50} | {baseline_result['score']:.4f} ({baseline_result['label'][:4]}) | {transformer_result['score']:.4f} ({transformer_result['label'][:4]})")

print("\n" + "=" * 60)
print("Testing complete!")
print("=" * 60)
