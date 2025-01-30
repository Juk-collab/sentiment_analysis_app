from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import requests
import json
from groq import Groq
import os

# Initialize FastAPI app
app = FastAPI()

# Load your Hugging Face model and tokenizer for the custom model (DistilBERT)
model_name = "Ahonenj/imdb-distilbert-finetuned"  # Replace with your model path
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Define the structure of the input data
class AnalyzeRequest(BaseModel):
    text: str
    model: str  # "custom" or "llama"

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float

# Define function to predict sentiment using the custom Hugging Face model
def predict_custom_model(text: str):
    # Tokenize and predict using the Hugging Face model
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    confidence, predicted_class = torch.max(probabilities, dim=-1)
    
    sentiment = "positive" if predicted_class == 1 else "negative"
    return sentiment, confidence.item()

# Define function to predict sentiment using Llama model from Groq Cloud
def predict_llama_model(text: str):

    api_key = os.environ.get("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    
    # Define the model endpoint for Llama 3 sentiment analysis
    model_endpoint = "llama3-70b-8192"
    
    # Prepare the request payload
    prompt = f"You must use sentiment_accuracy format e.g. negative_0.9124, positive_0.8523. Classify the sentiment of this text as positive or negative: '{text}'"
    messages = [
    {"role": "user", "content": prompt}  # The prompt for sentiment classification
]

    # Send the message and get the response
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model_endpoint,
        temperature=0.7,
        max_tokens=1000,
    )
    cont = chat_completion.choices[0].message.content
    sentiment, confidence = cont.split("_") 
    try:
        confidence = float(confidence)
    except:
        confidence = 0.00
    if "positive" in sentiment: 
        sentiment="positive"
    elif "negative in sentiment":
        sentiment="negative"
    return sentiment, confidence

# Define the /analyze endpoint
@app.post("/analyze/", response_model=SentimentResponse)
async def analyze(request: AnalyzeRequest):
    if request.model == "custom":
        sentiment, confidence = predict_custom_model(request.text)
    elif request.model == "llama":
        sentiment, confidence = predict_llama_model(request.text)
    else:
        raise HTTPException(status_code=400, detail="Invalid model specified. Use 'custom' or 'llama'.")
    
    return SentimentResponse(sentiment=sentiment, confidence=confidence)