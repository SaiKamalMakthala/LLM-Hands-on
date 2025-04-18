
import streamlit as st
import numpy as np
from keras.models import load_model
from utils.preprocess import get_sentiment_score, preprocess_input

# Load model
model = load_model('model/lstm_model.h5')

st.title("🧠 NBA Clutch Performance Predictor")
st.markdown("Predict if a player will perform clutch based on sentiment + stats.")

# User input
player = st.text_input("Enter Player Name", "Stephen Curry")
date = st.date_input("Select Game Date")
run = st.button("Predict")

if run:
    date_str = date.strftime('%Y-%m-%d')
    
    # Get sentiment score
    sentiment = get_sentiment_score(player, date_str)
    
    # Preprocess input for model
    X_input = preprocess_input(sentiment)  # shape must match LSTM input shape
    
    # Predict
    pred = model.predict(X_input)[0][0]
    label = "🔥 Clutch Performance" if pred > 0.5 else "😐 Not Clutch"
    
    st.metric("Sentiment Score", round(sentiment, 3))
    st.metric("Prediction Score", round(pred, 3))
    st.success(f"Prediction: {label}")
