import streamlit as st
import pickle

st.set_page_config(page_title="AI Resume Screening System")

st.title("AI Resume Screening System")
st.write("Paste a resume below and the system will predict the job category.")

try:
    model = pickle.load(open("resume_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    st.success("Model loaded successfully")
except Exception as e:
    st.error("Model loading error")
    st.write(e)

resume_text = st.text_area("Enter Resume Text")

if st.button("Predict Category"):
    if resume_text.strip() == "":
        st.warning("Please enter resume text.")
    else:
        text_vector = vectorizer.transform([resume_text])
        prediction = model.predict(text_vector)
        st.success("Predicted Job Category: " + prediction[0])