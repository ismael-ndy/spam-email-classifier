# Spam Email Classifier

This project provides a **Spam Email Classifier** using `Logistic Regression` and `Random Forest` models. The classifier leverages **TF-IDF vectorization** and **feature engineering** based on common spam words to detect potentially malicious or spam emails.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Training the Models](#training-the-models)
- [Evaluation Metrics](#evaluation-metrics)
- [Saving the Models](#saving-the-models)
- [Future Improvements](#future-improvements)

---

## Project Overview

The project consists of:
1. **Data Preprocessing**: Loads and preprocesses data, with TF-IDF vectorization and additional feature engineering.
2. **Model Training**: Trains both Logistic Regression and Random Forest models on the preprocessed data.
3. **Model Persistence**: Saves the trained models and TF-IDF vectorizer for future predictions.
4. **Evaluation**: Provides accuracy, precision, recall, and F1-score metrics.

## Features

- **Data Loading and Preprocessing**: Loads multiple CSV files, combines them, and extracts features relevant to spam classification.
- **Feature Engineering**: Counts occurrences of common spam words and performs TF-IDF vectorization.
- **Model Training**: Uses Logistic Regression and Random Forest models to classify emails.
- **Model Evaluation**: Calculates and displays performance metrics like accuracy, precision, recall, and F1-score.
- **Model Saving**: Saves the trained models and vectorizer for deployment in a web app.

## Dataset

The dataset used is a combination of various sources of labeled emails[^1].
It is stored in five separate CSV files in `./src/Dataset/`. These files are concatenated into a single DataFrame for processing. Each file should contain the following columns. 

- `text_combined`: The email text to classify.
- `label`: The label for each email (e.g., `spam` or `not spam`).


## Training the Models

The script `main.py`:
1. Loads and concatenates the dataset files.
2. Encodes labels using `LabelEncoder`.
3. Splits the dataset into training and testing sets.
4. Applies TF-IDF vectorization with unigram and bigram features.
5. Combines TF-IDF features with additional features (like spam word counts).
6. Trains and saves both the Logistic Regression and Random Forest models.

## Evaluation Metrics

Each model is evaluated using the following metrics:
- **Accuracy**: Measures the overall correctness.
- **Precision**: Indicates how many identified spam emails were actually spam.
- **Recall**: Shows how well the model identifies actual spam emails.
- **F1 Score**: Provides a balance between precision and recall.

The metrics are displayed in the console after training.

## Saving the Models

The trained models and TF-IDF vectorizer are saved to `./src/trained_models/`. The vectorizer file (`tfidf_model.pkl`) and each model file (e.g., `LogisticRegression_model.pkl`, `RandomForestClassifier_model.pkl`) are stored in this directory.

## Future Improvements

1. **Add More Features**: Experiment with additional feature engineering to capture more nuanced aspects of spam emails.
2. **Add More Models**: Test additional classifiers like SVM or Neural Networks.
3. **Hyperparameter Tuning**: Use `GridSearchCV` or `RandomizedSearchCV` for improved model performance.
4. **Optimize Web App**: Develop a user-friendly and responsive front end for the web app, and consider deploying it to a cloud platform like AWS or Heroku.

[^1]: Al-Subaiey, A., Al-Thani, M., Alam, N. A., Antora, K. F., Khandakar, A., & Zaman, S. A. U. (2024, May 19). Novel Interpretable and Robust Web-based AI Platform for Phishing Email Detection. ArXiv.org. https://arxiv.org/abs/2405.11619
