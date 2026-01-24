import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Page Setup

st.set_page_config(page_title="Email Spam Classifier", page_icon="📩", layout="centered")

# Simple Dark Theme CSS

st.markdown("""
<style>
.stApp {
    background-color: #0f172a;
    color: white;
}

h1, h2, h3 {
    color: white !important;
}

/* Button */
.stButton button {
    background: #2563eb !important;
    color: white !important;
    border-radius: 10px !important;
    padding: 10px 18px !important;
    font-weight: 600 !important;
}
.stButton button:hover {
    background: #1d4ed8 !important;
}

/* Result Box */
.result-box {
    background: #111827;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 18px;
    margin-top: 15px;
}
</style>
""", unsafe_allow_html=True)

# NLTK setup (avoid re-download)

@st.cache_resource
def download_nltk():
    nltk.download("stopwords")
    nltk.download("punkt")

download_nltk()

ps = PorterStemmer()

# Text preprocessing

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    # Keep only alphanumeric
    tokens = [t for t in tokens if t.isalnum()]

    # Remove stopwords + punctuation
    tokens = [t for t in tokens if t not in stopwords.words("english")]

    # Stemming
    tokens = [ps.stem(t) for t in tokens]

    return " ".join(tokens)

# Load model + vectorizer

tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# UI

st.title("📩 Email Spam Classifier")

st.markdown("### ✍️ Enter Email Text")
input_sms = st.text_area(
    "",
    height=150,
    placeholder="Paste your email message here...",
    label_visibility="collapsed"
)

predict = st.button("Predict Spam / Not Spam")

# Prediction (ONLY on button click)

if predict:
    if input_sms.strip() == "":
        st.warning("Please enter email text.")
    else:
        transformed = transform_text(input_sms)
        vector_input = tfidf.transform([transformed])
        result = model.predict(vector_input)[0]


        st.subheader("Result")

        if result == 1:
            st.error("🚨 SPAM EMAIL")
        else:
            st.success("✅ NOT SPAM (SAFE)")

        st.markdown("</div>", unsafe_allow_html=True)
