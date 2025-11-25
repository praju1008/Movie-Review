import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Parameters (must match your training)
vocab_size = 10000
max_len = 100

# Load the model and the IMDB word index
model = load_model("sentiment_model.h5")
word_index = imdb.get_word_index()

def preprocess_text(text: str):
    """Convert raw text to padded integer sequence as used by IMDB model."""
    words = text.lower().split()
    encoded_review = []
    for word in words:
        idx = word_index.get(word, 2)  # 2: OOV
        if idx >= vocab_size:
            idx = 2  # OOV
        encoded_review.append(idx)
    padded_review = sequence.pad_sequences([encoded_review], maxlen=max_len, padding="post", truncating="post")
    return np.array(padded_review)

def predict_sentiment(review: str):
    padded_review = preprocess_text(review)
    prediction = model.predict(padded_review)
    prob = float(prediction[0][0])
    sentiment = "Positive" if prob >= 0.5 else "Negative"
    return sentiment, prob

# Streamlit UI
st.title("Movie Review Sentiment Analysis")
st.write("Enter a movie review below to predict its sentiment (Positive/Negative).")

user_input = st.text_area("Movie Review", "")

if st.button("Predict Sentiment"):
    if user_input.strip():
        sentiment, prob = predict_sentiment(user_input)
        st.write(f"The predicted sentiment is: **{sentiment}** (score: {prob:.3f})")
    else:
        st.write("Please enter a movie review to analyze.")
