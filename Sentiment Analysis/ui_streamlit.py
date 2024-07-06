import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from joblib import load
import spacy
import time
from wordcloud import WordCloud

nlp = spacy.load("en_core_web_lg")

__model = None

# for preprocessing
def preprocess(text):
    doc = nlp(text)

    filtered_tokens = []

    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_tokens.append(token.lemma_)
    
    return " ".join(filtered_tokens)

# vectorise
def vectorize(text):
    return nlp(text).vector.reshape(1, -1)

# for finding the bias
def get_bias(text):
    clean_text = preprocess(text)
    embeddings = vectorize(clean_text)
    
    prediction = __model.predict(embeddings)[0]

    if prediction == -1:
        return "Negative"
    elif prediction == 1:
        return "Positive"
    else:
        return "Neutral"

# for loading the model
def load_saved_model():
    global __model
    
    with open("model.joblib", "rb") as f:
        __model = load(f)

# Main function to run the Streamlit app
def main():

    load_saved_model()

    st.title('Sentiment Analysis Web App')
    st.sidebar.title('Navigation')
    page = st.sidebar.radio('Go to', ('Home', 'Predict', 'About'))

    if page == 'Home':
        st.header('Home Page')
        st.write('Welcome to Sentiment Analysis Web App!')
        st.write('This app predicts sentiment (Positive, Negative, Neutral) based on user input.')

    elif page == 'Predict':
        st.header('Predict Sentiment')
        text = st.text_area('Enter your comment here:', height=200)
        if st.button('Predict'):
            prediction = get_bias(text)
            st.write(f'Predicted Sentiment: {prediction}')

    elif page == 'About':
        st.header('About')
        st.write('This is a sentiment analysis web application.')
        st.write('Created by Gaurav Singh Bisht.')

if __name__ == '__main__':
    main()
