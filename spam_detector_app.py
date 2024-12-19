import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import re
import nltk

# Download NLTK stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Function to clean and preprocess text
def clean_text(text):
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove non-alphabetic characters
    text = text.lower()  # Convert to lowercase
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Load your trained model and vectorizer
# (Ensure you save your model and vectorizer from your training script)
import pickle
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('spam_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

# Streamlit app interface
st.title("Spam SMS Detector")
st.write("Type a message below to check if it's Spam or Ham!")

# Input from the user
user_input = st.text_area("Enter the message:", help="Type an SMS or message to classify it as Spam or Ham.")

st.markdown(
    """
    <style>
    .title {
        font-size: 48px;
        font-weight: bold;
        color: #FF5733;
    }
    </style>
    """, unsafe_allow_html=True
)

if st.button("Check", key="check_button"):
    if user_input.strip():  # Check if input is not empty
        with st.spinner("Analyzing..."):
        # Simulate processing time
            import time
            time.sleep(1)
            cleaned_message = clean_text(user_input)  # Clean the message
            vectorized_message = vectorizer.transform([cleaned_message])  # Vectorize
            prediction = model.predict(vectorized_message)
            result = "Spam" if prediction[0] == 1 else "Ham"
        
            # Display with styled background
            if result == "Spam":
                st.markdown(
                    '<p style="background-color: #FF4C4C; padding: 10px; border-radius: 10px; color: white;">The message is Spam!</p>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<p style="background-color: #4CAF50; padding: 10px; border-radius: 10px; color: white;">The message is Ham!</p>',
                    unsafe_allow_html=True,
                )
    elif len(user_input) > 500:
        st.warning("⚠️ Message too long! Please enter less than 500 characters.")
    else:
        st.warning("⚠️ Please enter a valid message!")

st.markdown('<p class="title">Spam SMS Detector 📱</p>', unsafe_allow_html=True)
st.write("Analyze your text messages and determine if they're **Spam** or **Ham** instantly! 🚀")

st.sidebar.title("About the App")
st.sidebar.info(
    """
    This app uses a **Naive Bayes Classifier** to detect if an SMS is spam or ham.
    - Enter a message in the text box.
    - Click the "Check" button to see the result.
    """
)
st.sidebar.markdown("### Fun Fact")
st.sidebar.write("Did you know? The term **spam** originated from a Monty Python sketch where the word spam was repeated excessively!")




st.markdown(
    """
    ---
    Made by Csaba Erdős.
    """
)

