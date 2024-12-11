from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# Initialize FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

# Load the saved models (TF-IDF Vectorizer and Logistic Regression Model)
with open('Tfidf.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

with open('model.pkl', 'rb') as f:
    prediction_model = pickle.load(f)

# Define the input data structure
class AnalyzePostRequest(BaseModel):
    tweet: str
    keywords: str
    statement: str = None
    label_probability: str = "Disagree"  # Default to "Disagree"

@app.post("/analyze_post")
async def analyze_post(request: AnalyzePostRequest):
    # Concatenate statement, keywords, and tweet
    statement = request.statement if request.statement else ""
    concatenated_text = f"{statement} {request.keywords} {request.tweet}"
    
    # Convert label probability: "Agree" -> 1.0, "Disagree" -> 0.0
    label_probability = 1.0 if request.label_probability.lower() == "agree" else 0.0

    # Transform the concatenated text using the saved TF-IDF model
    tfidf_features = tfidf_vectorizer.transform([concatenated_text])

    # Prepare the combined feature for prediction (TF-IDF features + '5_label_majority_answer')
    # Create a DataFrame from the TF-IDF array and add '5_label_majority_answer'
    X_combined = pd.DataFrame(tfidf_features.toarray())
    
    # In this case, we assume that '5_label_majority_answer' is part of the request for prediction
    # If you need to handle it dynamically (e.g., during model training), we will add it directly
    # Here, we are using the value of `label_probability` to simulate the presence of this feature
    X_combined['5_label_majority_answer'] = label_probability
    X_combined.columns = X_combined.columns.astype(str)

    # Make the prediction using the loaded prediction model
    prediction = prediction_model.predict(X_combined)
    
    # Return the prediction as a JSON response
    return {"prediction": int(prediction[0]), "label_probability": label_probability}
