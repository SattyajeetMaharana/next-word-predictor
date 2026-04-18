# ============================================================
# predict.py — Next Word Prediction Logic
# Autocorrect Keyboard System
# ============================================================
# This module:
#   - Loads the trained N-gram model
#   - Accepts a sentence as input
#   - Returns top-N predicted next words
# ============================================================

import pickle
import re
from collections import Counter


# ── Load Model ───────────────────────────────────────────────
def load_model(path="model/ngram_model.pkl"):
    """Load bigram and trigram models from disk."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["bigram"], data["trigram"]


# ── Preprocess Input ─────────────────────────────────────────
def clean_input(text):
    """Lowercase and remove punctuation from user input."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()


# ── Predict Next Word ────────────────────────────────────────
def predict_next_word(sentence, bigram_model, trigram_model, top_n=3):
    """
    Given an input sentence, predict the most probable next words.

    Strategy:
      1. Try trigram first (last 2 words as context)
      2. Fall back to bigram  (last 1 word as context)
      3. If nothing found, return empty list

    Parameters:
        sentence      (str)  : Input text typed by user
        bigram_model  (dict) : Trained bigram frequency dict
        trigram_model (dict) : Trained trigram frequency dict
        top_n         (int)  : Number of suggestions to return

    Returns:
        List of (word, probability) tuples
    """
    tokens = clean_input(sentence)

    if not tokens:
        return []

    predictions = Counter()

    # ── Try Trigram ───────────────────────────────────────────
    if len(tokens) >= 2:
        context = (tokens[-2], tokens[-1])
        if context in trigram_model:
            predictions.update(trigram_model[context])

    # ── Try Bigram (fallback / combine) ──────────────────────
    context = (tokens[-1],)
    if context in bigram_model:
        # Add bigram counts with lower weight so trigram takes priority
        for word, count in bigram_model[context].items():
            predictions[word] += count * 0.5   # half weight for bigram

    if not predictions:
        return []

    # ── Compute Probability ───────────────────────────────────
    total = sum(predictions.values())
    results = [
        (word, round(count / total * 100, 1))
        for word, count in predictions.most_common(top_n)
    ]
    return results


# ── CLI Demo ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  Next Word Predictor — CLI Mode")
    print("=" * 50)

    bigram_model, trigram_model = load_model()

    while True:
        user_input = input("\nEnter text (or 'quit' to exit): ").strip()
        if user_input.lower() == "quit":
            break
        if not user_input:
            print("Please enter some text.")
            continue

        suggestions = predict_next_word(user_input, bigram_model, trigram_model)

        if suggestions:
            print("\n📌 Top Predictions:")
            for i, (word, prob) in enumerate(suggestions, 1):
                print(f"   {i}. {word:20s}  ({prob}%)")
        else:
            print("⚠️  No predictions found for this input.")
