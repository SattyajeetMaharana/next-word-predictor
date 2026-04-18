# ============================================================
# app.py — Autocorrect Keyboard UI (Streamlit)
# Next Word Prediction System
# ============================================================
# Run with:  streamlit run app.py
# ============================================================

import os
import streamlit as st
from predict import load_model, predict_next_word

# ── Page Configuration ────────────────────────────────────────
st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="⌨️",
    layout="centered"
)

# ── Custom CSS Styling ────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Header ── */
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #6c757d;
        font-size: 1rem;
    }

    /* ── Suggestion Cards ── */
    .suggestion-card {
        background: linear-gradient(135deg, #667eea15, #764ba215);
        border: 1.5px solid #667eea44;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        text-align: center;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.1rem;
        font-weight: 700;
        color: #4a3f8f;
        transition: all 0.2s;
        cursor: pointer;
    }
    .suggestion-card:hover {
        background: linear-gradient(135deg, #667eea30, #764ba230);
        border-color: #667eea;
        transform: translateY(-2px);
    }
    .prob-badge {
        font-size: 0.75rem;
        font-weight: 400;
        color: #9b8fc7;
        display: block;
        margin-top: 0.2rem;
    }

    /* ── Info Box ── */
    .info-box {
        background: #f8f9ff;
        border-left: 4px solid #667eea;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1.2rem;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: #555;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        color: #aaa;
        font-size: 0.8rem;
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Load Model ────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading prediction model...")
def get_model():
    """Load model once and cache it for the session."""
    model_path = "model/ngram_model.pkl"
    if not os.path.exists(model_path):
        return None, None
    return load_model(model_path)


bigram_model, trigram_model = get_model()


# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>⌨️ Next Word Predictor</h1>
    <p>Autocorrect Keyboard System using N-gram Language Model</p>
</div>
""", unsafe_allow_html=True)

st.divider()


# ── Model Status ──────────────────────────────────────────────
if bigram_model is None:
    st.error("""
    ⚠️ **Model not found!**
    Please train the model first by running:
    ```
    python train_model.py
    ```
    """)
    st.stop()
else:
    st.success("✅ Model loaded successfully and ready to predict!")


# ── Input Section ─────────────────────────────────────────────
st.markdown("### 📝 Type Your Sentence")

user_input = st.text_input(
    label="Input",
    placeholder="e.g.  I am going to ...",
    label_visibility="collapsed"
)

col_btn, col_clear = st.columns([1, 5])
with col_btn:
    predict_btn = st.button("Predict ➜", use_container_width=True, type="primary")


# ── Prediction Output ─────────────────────────────────────────
if predict_btn or user_input:
    if not user_input.strip():
        st.warning("Please enter some text first.")
    else:
        suggestions = predict_next_word(
            user_input, bigram_model, trigram_model, top_n=3
        )

        st.markdown("### 💡 Suggested Next Words")

        if suggestions:
            cols = st.columns(len(suggestions))
            for col, (word, prob) in zip(cols, suggestions):
                with col:
                    st.markdown(f"""
                    <div class="suggestion-card">
                        {word}
                        <span class="prob-badge">{prob}% probability</span>
                    </div>
                    """, unsafe_allow_html=True)

            # ── Full Sentence Preview ──────────────────────────
            st.markdown("#### ✏️ Sentence Preview")
            for word, _ in suggestions:
                st.code(f"{user_input.strip()} {word}", language=None)

        else:
            st.info("🔍 No predictions found. Try a different phrase or add more training data.")


# ── How It Works Section ──────────────────────────────────────
with st.expander("ℹ️ How does this work?"):
    st.markdown("""
    This system uses **N-gram Language Modeling** — a classic NLP technique:

    | Model     | Context Used          | Example                         |
    |-----------|-----------------------|---------------------------------|
    | Bigram    | Last **1** word       | `going` → `to`                  |
    | Trigram   | Last **2** words      | `am going` → `to`               |

    **Steps:**
    1. A large text corpus is loaded and cleaned
    2. Word sequences (n-grams) are counted
    3. Probability = count of sequence / total occurrences of context
    4. Top-3 highest probability words are returned as suggestions

    **Technologies used:**
    - Python · NLTK · Pickle · Streamlit
    """)


# ── Footer ────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Next Word Predictor · Autocorrect Keyboard System · Built with Python & Streamlit
</div>
""", unsafe_allow_html=True)
