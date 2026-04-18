# ⌨️ Next Word Predictor — Autocorrect Keyboard System

An intelligent autocorrect keyboard that predicts the next word in a sentence using **N-gram Language Modeling**. Built with Python, NLTK, and Streamlit.

---

## 📌 Project Description

This system anticipates the next word a user is likely to type by analyzing the context of preceding words. It uses **Bigram** and **Trigram** models trained on a text corpus to calculate word probabilities and suggest the top 3 most likely next words.

---

## 🚀 Features

- ✅ Next word prediction using N-gram language model
- ✅ Bigram + Trigram hybrid model for better accuracy
- ✅ Text preprocessing (lowercase, punctuation removal)
- ✅ Probability-based word ranking
- ✅ Interactive web UI built with Streamlit
- ✅ Top-3 word suggestions with probability scores
- ✅ Sentence preview with suggested word

---

## 🛠️ Technologies Used

| Technology | Purpose                        |
|------------|--------------------------------|
| Python     | Core programming language      |
| NLTK       | Natural Language Processing    |
| Pickle     | Model serialization            |
| Streamlit  | Web application interface      |
| N-gram     | Language modeling technique    |

---

## 📁 Project Structure

```
next-word-predictor/
│
├── dataset/
│   └── text_corpus.txt       # Training text data
│
├── model/
│   └── ngram_model.pkl       # Trained model (auto-generated)
│
├── train_model.py            # Train and save the N-gram model
├── predict.py                # Prediction logic + CLI mode
├── app.py                    # Streamlit web application
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## ⚙️ How to Run

### Step 1 — Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/next-word-predictor.git
cd next-word-predictor
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Train the model
```bash
python train_model.py
```

### Step 4 — Run the web app
```bash
streamlit run app.py
```

### (Optional) Run CLI mode
```bash
python predict.py
```

---

## 🧠 How It Works

1. **Load Corpus** — Read the training text from `dataset/text_corpus.txt`
2. **Preprocess** — Convert to lowercase, remove punctuation, tokenize
3. **Build N-grams** — Count Bigram and Trigram frequencies
4. **Predict** — Given input text, find matching context and return top-N words by probability
5. **Display** — Show suggestions in the Streamlit UI

### Prediction Strategy

| Model   | Context          | Priority |
|---------|------------------|----------|
| Trigram | Last 2 words     | High     |
| Bigram  | Last 1 word      | Fallback |

---

## 📊 Example

**Input:**
```
I am going to
```

**Output:**
```
1. school     (45.2%)
2. the        (30.1%)
3. market     (24.7%)
```

---

## 📚 Dataset

The model is trained on `dataset/text_corpus.txt`. You can replace or extend this file with any larger text corpus (Wikipedia, books, dialogues) to improve prediction accuracy.

---

## 👨‍💻 Author

Built as part of a Computer Science project on Natural Language Processing.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
