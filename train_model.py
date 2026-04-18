# ============================================================
# train_model.py — Next Word Predictor (N-gram Model)
# Autocorrect Keyboard System
# ============================================================
# This script:
#   1. Loads a text corpus
#   2. Preprocesses the text (lowercase, clean punctuation)
#   3. Builds Bigram + Trigram frequency models
#   4. Saves the trained model as model/ngram_model.pkl
# ============================================================

import os
import re
import pickle
from collections import defaultdict, Counter

# ── 1. Load Dataset ──────────────────────────────────────────
def load_corpus(filepath):
    """Read all lines from the text corpus."""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"[✓] Corpus loaded — {len(text)} characters")
    return text


# ── 2. Preprocess Text ───────────────────────────────────────
def preprocess(text):
    """
    Clean raw text:
      - Convert to lowercase
      - Remove special characters (keep letters, digits, spaces)
      - Split into list of word tokens
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)   # remove punctuation
    tokens = text.split()
    print(f"[✓] Preprocessing done — {len(tokens)} tokens")
    return tokens


# ── 3. Build N-gram Models ───────────────────────────────────
def build_ngram_model(tokens):
    """
    Build two frequency dictionaries:
      bigram_model  : {(w1,)        : Counter({next_word: count})}
      trigram_model : {(w1, w2)     : Counter({next_word: count})}
    """
    bigram_model  = defaultdict(Counter)
    trigram_model = defaultdict(Counter)

    for i in range(len(tokens) - 1):
        w1 = tokens[i]
        w2 = tokens[i + 1]
        bigram_model[(w1,)][w2] += 1          # bigram

    for i in range(len(tokens) - 2):
        w1, w2, w3 = tokens[i], tokens[i+1], tokens[i+2]
        trigram_model[(w1, w2)][w3] += 1      # trigram

    print(f"[✓] Bigram  model — {len(bigram_model)} unique contexts")
    print(f"[✓] Trigram model — {len(trigram_model)} unique contexts")
    return bigram_model, trigram_model


# ── 4. Save Model ────────────────────────────────────────────
def save_model(bigram_model, trigram_model, path="model/ngram_model.pkl"):
    """Serialize both models into a single .pkl file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"bigram": bigram_model, "trigram": trigram_model}, f)
    print(f"[✓] Model saved → {path}")


# ── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    corpus_path = "dataset/text_corpus.txt"

    text          = load_corpus(corpus_path)
    tokens        = preprocess(text)
    bigram, trigram = build_ngram_model(tokens)
    save_model(bigram, trigram)

    print("\n✅ Training complete! Model saved to model/ngram_model.pkl")
