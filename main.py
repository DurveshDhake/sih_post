from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os
import sys
import requests

# Initialize FastAPI app
app = FastAPI()
# List of origins that are allowed to make requests
origins = [
    "http://localhost:4200",  # Allow requests from Angular frontend
    # You can add more allowed origins if needed
]

# Add CORSMiddleware to handle cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Specify the allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Check and download model files if missing
model_files = ['tfidf.pkl', 'model.pkl']
model_urls = {
    'tfidf.pkl': 'https://example.com/path-to-tfidf.pkl',
    'model.pkl': 'https://example.com/path-to-model.pkl'
}


for file in model_files:
    file_path = os.path.join(BASE_DIR, file)
    if not os.path.exists(file_path):
        if file in model_urls:
            try:
                print(f"Downloading {file}...")
                response = requests.get(model_urls[file])
                response.raise_for_status()
                with open(file_path, 'wb') as f:
                    f.write(response.content)
            except requests.exceptions.RequestException as e:
                sys.exit(f"Error downloading {file}: {e}")
        else:
            sys.exit(f"Error: {file} is missing and no URL is provided.")

# Load the saved models (TF-IDF Vectorizer and Logistic Regression Model)
try:
    with open(os.path.join(BASE_DIR, 'tfidf.pkl'), 'rb') as f:
        tfidf_vectorizer = pickle.load(f)

    with open(os.path.join(BASE_DIR, 'model.pkl'), 'rb') as f:
        prediction_model = pickle.load(f)
except FileNotFoundError as e:
    sys.exit(f"Error loading model files: {e}")
except Exception as e:
    sys.exit(f"Unexpected error loading model files: {e}")

# Define the input data structure
class AnalyzePostRequest(BaseModel):
    tweet: str
    keywords: str
    statement: str = None
    label_probability: str = "Disagree"  # Default to "Disagree"

@app.post("/analyze_post")
async def analyze_post(request: AnalyzePostRequest):
    try:
        # Concatenate statement, keywords, and tweet
        statement = request.statement if request.statement else ""
        concatenated_text = f"{statement} {request.keywords} {request.tweet}"

        # Convert label probability: "Agree" -> 1.0, "Disagree" -> 0.0
        label_probability = 1.0 if request.label_probability.lower() == "agree" else 0.0

        # Transform the concatenated text using the saved TF-IDF model
        tfidf_features = tfidf_vectorizer.transform([concatenated_text])

        # Prepare the combined feature for prediction (TF-IDF features + '5_label_majority_answer')
        X_combined = pd.DataFrame(tfidf_features.toarray())

        # Add '5_label_majority_answer' to the features
        X_combined['5_label_majority_answer'] = label_probability
        X_combined.columns = X_combined.columns.astype(str)

        # Make the prediction using the loaded prediction model
        prediction = prediction_model.predict(X_combined)

        # Return the prediction as a JSON response
        return {"prediction": int(prediction[0]), "label_probability": label_probability}

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {e}")

# Root endpoint for basic health check
@app.get("/")
def root():
    return {"message": "TF-IDF Vectorizer and Prediction Model are ready to process text."}
