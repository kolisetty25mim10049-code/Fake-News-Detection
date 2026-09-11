# 📰 Fake News Detection System

## 📌 Project Overview

The Fake News Detection System is an Artificial Intelligence and Machine Learning based web application designed to classify a given news article as likely **REAL** or **FAKE**.

The system uses Natural Language Processing (NLP) techniques to convert news text into numerical features using **TF-IDF Vectorization** and then uses a **Logistic Regression** machine learning model to make the prediction.

The application provides a simple and user-friendly interface built using **Streamlit**.

---

## 🎯 Objectives

- Detect potentially fake news articles.
- Apply Natural Language Processing to news text.
- Convert text into numerical features using TF-IDF.
- Train a Logistic Regression classification model.
- Provide predictions through an interactive web interface.
- Display the prediction and confidence score to the user.

---

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Joblib
- Streamlit
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Logistic Regression

---

## 📂 Project Structure

```text
Fake-News-Detection/
│
├── src/
│   ├── data/
│   │   └── news_dataset.csv
│   │
│   ├── app.py
│   ├── train_model.py
│   ├── fake_news_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── .gitignore
└── README.md
