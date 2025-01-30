import requests

# Define the URL for the FastAPI application
url = "http://127.0.0.1:8000/analyze/"  # Change to your server's URL if needed

# Function to test the sentiment analysis for the custom Hugging Face model
def test_custom_model(text):
    payload = {
        "text": text,
        "model": "custom"
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        result = response.json()
        print(f"\nText: {text}")
        print("Custom Model Response:")
        print(f"Sentiment: {result['sentiment']}, Confidence: {result['confidence']:.4f}")
    else:
        print(f"Error: {response.status_code}, Message: {response.text}")

# Function to test the sentiment analysis for the Llama model from Groq
def test_llama_model(text):
    payload = {
        "text": text,
        "model": "llama"
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        result = response.json()
        print(f"Text: {text}")
        print("\nLlama Model Response:")
        print(f"Sentiment: {result['sentiment']}, Confidence: {result['confidence']:.4f}")
    else:
        print(f"Error: {response.status_code}, Message: {response.text}")

# Example texts to test
text_1 = "I liked the movie very much."
text_2 = "The movie was a complete disaster."

# Test both models with the sample texts
test_custom_model(text_1)
test_custom_model(text_2)

test_llama_model(text_1)
test_llama_model(text_2)