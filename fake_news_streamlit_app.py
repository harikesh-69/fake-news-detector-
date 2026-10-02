import re
import string
import pandas as pd
import streamlit as st

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---------------------------------------------------
# Page config
# ---------------------------------------------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# ---------------------------------------------------
# One-time NLTK downloads (cached so it only runs once)
# ---------------------------------------------------
@st.cache_resource
def setup_nltk():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    return True

setup_nltk()

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# ---------------------------------------------------
# Load data + train model once, then cache it
# (so it doesn't retrain every time you click Predict)
# ---------------------------------------------------
@st.cache_resource
def load_and_train():
    url = "https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv"
    df = pd.read_csv(url)
    df = df[['text', 'label']]
    df['clean_text'] = df['text'].apply(preprocess_text)

    X = df['clean_text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    test_accuracy = accuracy_score(y_test, model.predict(X_test_tfidf))

    return tfidf, model, test_accuracy, len(df)

# ---------------------------------------------------
# UI
# ---------------------------------------------------
st.title("📰 Fake News Detection")
st.caption("NLP mini project — TF-IDF + Logistic Regression | Harikesh Sharma (Roll No. 36)")

with st.spinner("Loading dataset and training model (first run only)..."):
    tfidf, model, test_accuracy, dataset_size = load_and_train()

st.success(f"Model ready — trained on {dataset_size:,} articles | Test Accuracy: {test_accuracy*100:.2f}%")

st.divider()

st.subheader("Try it out")

sample_options = {
    "-- Select a sample article --": "",
    "Sample 1 (tends to score REAL)": "The Federal Reserve announced today that it will maintain current interest rates following its policy meeting, citing stable inflation data and steady employment figures across the country.",
    "Sample 2 (tends to score FAKE)": "You won't believe what scientists just discovered! This one weird trick doctors don't want you to know is shocking the entire nation and mainstream media refuses to cover it.",
}
choice = st.selectbox("Quick test with a sample, or paste your own below:", list(sample_options.keys()))

default_text = sample_options[choice]
user_input = st.text_area("News article text:", value=default_text, height=180,
                           placeholder="Paste a news article or headline here...")

col1, col2 = st.columns([1, 3])
with col1:
    predict_clicked = st.button("🔍 Predict", type="primary", use_container_width=True)

if predict_clicked:
    if not user_input.strip():
        st.warning("Please enter or select some text first.")
    else:
        cleaned = preprocess_text(user_input)
        vec = tfidf.transform([cleaned])
        prediction = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        classes = model.classes_
        confidence = max(proba) * 100

        st.divider()
        if prediction == "FAKE":
            st.error(f"### Prediction: 🚫 FAKE  \nConfidence: {confidence:.2f}%")
        else:
            st.success(f"### Prediction: ✅ REAL  \nConfidence: {confidence:.2f}%")

        with st.expander("See prediction probabilities"):
            for cls, p in zip(classes, proba):
                st.write(f"{cls}: {p*100:.2f}%")

        with st.expander("See preprocessed text (what the model actually saw)"):
            st.code(cleaned)

st.divider()
st.caption("Dataset: McIntire Fake and Real News Dataset (6,335 articles) · Model: TF-IDF + Logistic Regression")
