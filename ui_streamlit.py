import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from joblib import load
from sklearn.metrics import confusion_matrix
import spacy
import time
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nlp = spacy.load("en_core_web_lg")

# Global variable for the model
__model = None

# Function to preprocess text
def preprocess(text):
    doc = nlp(text)
    filtered_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(filtered_tokens)

# Function to vectorize text
def vectorize(text):
    return nlp(text).vector.reshape(1, -1)

# Function to predict sentiment
def predict_sentiment(text):
    clean_text = preprocess(text)
    #print("Artist:",songs['artist'][num],"\nSong:",songs['song'][num],"\n")
    sid = SentimentIntensityAnalyzer()
    pscore=sid.polarity_scores(clean_text)
    prediction = "Positivity Score:"+ str(pscore['pos']*100)+"\nNegativity Score:" + str(pscore['neg']*100)+"\nNeutral Score:"+str(pscore['neu']*100)

    return prediction

    # if prediction == -1:
    #     return "Negative"
    # elif prediction == 1:
    #     return "Positive"
    # else:
    #     return "Neutral"

# Function to load the saved model
def load_model():
    global __model
    __model = load("model.joblib")

# Custom CSS for background and formatting
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
        color: #333333;
        font-family: 'Arial', sans-serif;
    }
    h1, h2, h3 {
        color: #d32f2f; /* Red */
    }
    .stButton button {
        background-color: #d32f2f; /* Red */
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #b71c1c; /* Darker Red */
    }
    .content-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .stTextInput input {
        background-color: #ffffff;
        border: 1px solid #ccc;
        padding: 10px;
        width: 100%;
        border-radius: 5px;
        box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.1);
    }
    .stTextInput input:focus {
        border-color: #1976d2; /* Blue */
        box-shadow: 0 0 5px rgba(25, 118, 210, 0.5); /* Blue */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main function to run the Streamlit app
def main():
    load_model()

    st.title('Sentiment Analysis Web App')

    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.write('Welcome to the Sentiment Analysis Web App! This app predicts sentiment (Positive, Negative, Neutral) based on user input.')
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.header('Predict Sentiment')
    text = st.text_area('Enter your comment here:', height=200)
    if st.button('Predict'):
        if text:
            prediction = predict_sentiment(text)
            st.markdown(
                f'<div style="font-size: 20px; font-weight: bold;">Predicted Sentiment: '
                f'<span style="color: #d32f2f;">{prediction}</span></div>',
                unsafe_allow_html=True
            )
        else:
            st.warning('Please enter a comment to predict.')
    st.markdown('</div>', unsafe_allow_html=True)
   
    st.markdown('</div>', unsafe_allow_html=True)

    st.header('About')
    st.write('This is a sentiment analysis web application.')
    st.write('Created by Vyomesh Naithani.')
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()