import pandas as pd
import os
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# Dataset path
DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "data",
    "news_dataset.csv"
)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Remove empty rows
df = df.dropna()

# Input and output
X = df["text"]
y = df["label"]

# Convert text into numerical TF-IDF features
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X_tfidf = vectorizer.fit_transform(X)

# Create Machine Learning model
model = LogisticRegression(
    max_iter=1000
)

# Train model
model.fit(X_tfidf, y)

# Save model
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "fake_news_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    os.path.dirname(__file__),
    "tfidf_vectorizer.pkl"
)

joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)

print("Model trained successfully!")
print("Model saved as:", MODEL_PATH)
print("Vectorizer saved as:", VECTORIZER_PATH)