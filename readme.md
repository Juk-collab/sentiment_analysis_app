# Sentiment Analysis App

## Overview

This project is a **Sentiment Analysis Web App** that allows users to analyze the sentiment of a given text using either:

- A **custom fine-tuned DistilBERT model** (Hugging Face)
- A **Llama 3 model** (via Groq API)

The frontend is built using **Streamlit**, while the backend is powered by **FastAPI**.

Note that the code has been tested to work only in Windows machine.

## Features

- **Text Input:** Users can enter any text for sentiment analysis.
- **Model Selection:** A dropdown allows users to choose between:
  - `custom` (Fine-tuned DistilBERT model)
  - `llama` (Llama 3 model via Groq API)
- **Analyze Sentiment Button:** Sends the request to the backend API.
- **Result Display:** Shows the **sentiment (positive/negative)** and the **confidence score**.
Note that the confidence of Llama3 is based on prompting, meaning the model decides itself how accurate it is.
## Technologies & Libraries Used

### Backend (FastAPI)

- `fastapi` - For building the API
- `uvicorn` - ASGI server for running FastAPI
- `torch` - PyTorch for handling the DistilBERT model
- `transformers` - Hugging Face Transformers library for loading the model
- `requests` - For making API calls to the Groq cloud service
- `pydantic` - For data validation in FastAPI

### Frontend (Streamlit)

- `streamlit` - For creating the interactive user interface

## Installation & Setup

### 1. Clone the Repository

```sh
git clone https://github.com/your-username/sentiment-analysis-app.git
cd sentiment-analysis-app
```

### 2. Install Dependencies

```sh
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

 **Groq API Key** can be set directly in your terminal:

```sh
export GROQ_API_KEY=your_api_key_here  # For Linux/macOS
set GROQ_API_KEY=your_api_key_here    # For Windows (CMD)
```

### 4. Run the Backend (FastAPI)

```sh
uvicorn sentiment_app:app --reload
```

*Default URL: *[*http://localhost:8000/*](http://localhost:8000/)


After backend is running you can run test.py, which tests both models with example text. 

### 5. Run the Frontend (Streamlit)

```sh
streamlit run frontend.py
```

*Default URL: *[*http://localhost:8501/*](http://localhost:8501/)

## API Endpoint

- **POST **``
  - **Request Body:**
    ```json
    {
      "text": "The movie was fantastic!",
      "model": "llama"
    }
    ```
  - **Response:**
    ```json
    {
      "sentiment": "positive",
      "confidence": 0.8523
    }
    ```

## How It Works

1. The user enters text and selects a model in the Streamlit UI.
2. The request is sent to the FastAPI backend.
3. The backend:
   - Uses **DistilBERT finetuned for sentiment analysis with IMDB dataset** if `model=custom`
   - Calls **Llama 3 via Groq API** if `model=llama`
4. The sentiment and confidence score are returned and displayed in the UI.