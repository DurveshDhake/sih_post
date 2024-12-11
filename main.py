from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import os  # Import missing os module
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Initialize FastAPI app
app = FastAPI()

# Fix the issue with BASE_DIR by using the correct name for the file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Use __file__ instead of _file_

# Load the saved models (TF-IDF Vectorizer and Logistic Regression Model)
try:
    with open(os.path.join(BASE_DIR, 'Tfidf.pkl'), 'rb') as f:
        tfidf_vectorizer = pickle.load(f)

    with open(os.path.join(BASE_DIR, 'model.pkl'), 'rb') as f:
        prediction_model = pickle.load(f)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error loading models: {str(e)}")

# Define a simple route for checking if the server is up
@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

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

        # Add the label_probability to the features dataframe
        X_combined['5_label_majority_answer'] = label_probability
        X_combined.columns = X_combined.columns.astype(str)  # Ensure columns are strings

        # Make the prediction using the loaded prediction model
        prediction = prediction_model.predict(X_combined)

        # Return the prediction as a JSON response
        return {"prediction": int(prediction[0]), "label_probability": label_probability}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in prediction: {str(e)}")
