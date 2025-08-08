import joblib
import streamlit as st

vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr.jb")


st.title("Fake News Detector")
st.write("Enter a News Article below to check whether it is Fake or Real. ")


news_input = st.text_area("News Article:","")

if st.button("Check News"):
    if news_input.strip():
        transform_input = vectorizer.transform([news_input])
        prediction = model.predict(transform_input)

        if prediction == 1:
            st.success("The News is Real! ")
        else:
            st.error("The News is Fake! ")
    else:
        st.warning("please enter some text to analyze. ")