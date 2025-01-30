import streamlit as st
import requests
import os

# Set the backend API URL (assumes FastAPI is running locally)
API_URL = "http://localhost:8000/analyze/"

# Streamlit Page Configuration
st.set_page_config(page_title="Sentiment Analysis", layout="wide")

# Page title
st.title("Sentiment Analysis App")

# Text input for user input
text_input = st.text_area("Enter text for sentiment analysis:", height=150)

# Dropdown for model selection
model = st.selectbox("Choose Model", ("custom", "llama"))

# Button to trigger the sentiment analysis
if st.button("Analyze Sentiment"):
    # Validate input text
    if text_input.strip() == "":
        st.error("Please enter some text.")
    else:
        # Prepare the request payload
        payload = {
            "text": text_input,
            "model": model
        }

        # Send the POST request to the FastAPI backend
        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()  # Will raise an HTTPError for bad responses

            # Parse the response
            sentiment_data = response.json()
            sentiment = sentiment_data["sentiment"]
            confidence = sentiment_data["confidence"]

            # Display the result
            st.success(f"Sentiment: {sentiment}")
            st.write(f"Confidence: {confidence:.4f}")

        except requests.exceptions.RequestException as e:
            st.error(f"Error: {e}")