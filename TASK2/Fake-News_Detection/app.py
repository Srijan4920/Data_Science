import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Streamlit App
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detection App")

user_input = st.text_area("Enter News Content", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some news text.")
    else:
        vec_input = vectorizer.transform([user_input])
        prediction = model.predict(vec_input)[0]
        label = "🟢 Real News" if prediction == 1 else "🔴 Fake News"
        st.subheader(f"Prediction: {label}")
