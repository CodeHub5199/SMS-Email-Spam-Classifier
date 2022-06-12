import streamlit as st
import pickle
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import  PorterStemmer
import string

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    # Removing Special Characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    # Removing Stopwords and Punctuations
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(ps.stem(i))

    return ' '.join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title('Email/SMS Spam Classifier')

input_sms = st.text_area('Enter the message')

if st.button('Predict'):
    # Preprocessing
    transformed_msg = transform_text(input_sms)

    # Vectorizing
    vector_input = tfidf.transform([transformed_msg])

    # Predict
    result = model.predict(vector_input)[0]

    # Display
    if result == 0:
        st.header('Not Spam')
    else:
        st.header('Spam')
