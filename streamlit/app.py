import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string


ps = PorterStemmer()
tfidf = pickle.load(open('vectorizer.pkl' , 'rb'))
model = pickle.load(open('model.pkl' , 'rb'))

def transform_text(text):
    
#   convert into lowercase
    text = text.lower()
    
#   convert sentence in list of words
    text = nltk.word_tokenize(text)
    
#   removal of special characters
    y = []
    for word in text:
        if word.isalnum():
            y.append(word)
    
    text = y.copy()
    y.clear()
    
#   removal of stopwords and punctuation
    for word in text:
        if word not in stopwords.words('english') and word not in string.punctuation:
            y.append(word)

    text = y.copy()
    y.clear()
    
#   Stemming
    for word in text:
        y.append(ps.stem(word))
        
    return " ".join(y)

# ----------------------------------------------------------------------------------------------------

st.title("SMS Spam Classifier")
input_sms  = st.text_area("Enter your SMS")

if st.button("Predict") and input_sms:
    transform_sms = transform_text(input_sms)
    vector_sms  = tfidf.transform([transform_sms])
    result = model.predict(vector_sms)[0]
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
    
