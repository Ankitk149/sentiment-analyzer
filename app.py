import streamlit as st
from predict import predict

st.set_page_config(page_title="Sentiment Analyzer", page_icon="🎬")

st.title("🎬 IMDB Sentiment Analyzer")

st.markdown("""
Analyze movie reviews using a Deep Learning model.

Type a review below or try the example buttons.
""")

st.write("Try an example:")

col1, col2 = st.columns(2)

with col1:
    if st.button("Positive Example"):
        user_input = "This movie was amazing and fantastic"
        st.write(user_input)

with col2:
    if st.button("Negative Example"):
        user_input = "This was the worst movie I have ever seen"
        st.write(user_input)

user_input = st.text_area("Enter your review:")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        sentiment, confidence = predict(user_input)

        if sentiment == "Positive":
            st.success(f"Positive 😊 (Confidence: {confidence:.2f})")
        else:
            st.error(f"Negative 😞 (Confidence: {confidence:.2f})")
        st.progress(int(confidence * 100))