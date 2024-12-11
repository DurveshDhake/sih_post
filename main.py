from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

# Initialize FastAPI app
app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the saved models (TF-IDF Vectorizer and Logistic Regression Model)
with open(os.path.join(BASE_DIR, 'Tfidf.pkl'), 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

with open(os.path.join(BASE_DIR, 'model.pkl'), 'rb') as f:
    prediction_model = pickle.load(f)

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
