import streamlit as st
import pickle

def clean_text(text):
    return text.strip().lower()

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Fake Job Predictor")

input_job_detail = st.text_area("Enter the job detail")

if st.button('Result'):
    #1. preprocess
    transform_input = clean_text(input_job_detail)
    # 2. vectorizer
    vector_input = tfidf.fit_transform([transform_input])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. display
    if result == 1:
        st.header("Fake Job")
    else:
        st.header("Real Job")