# Spam Email Classifier

This project provides a **Spam Email Classifier** using `Logistic Regression` and `Random Forest` models. The classifier leverages **TF-IDF vectorization** and **feature engineering** based on common spam words to detect potentially malicious or spam emails. Additionally, it includes a web application built with Flask that detects if the input email is spam and, if it is not, uses a self-hosted Llama 3.2 model with Ollama to generate a suggested response.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Training the Models](#training-the-models)
- [Evaluation Metrics](#evaluation-metrics)
- [Saving the Models](#saving-the-models)
- [Web Application](#web-application)
- [Setup Instructions](#setup-instructions)
- [Future Improvements](#future-improvements)

---

## Project Overview

The project consists of:
1. **Data Preprocessing**: Loads and preprocesses data, with TF-IDF vectorization and additional feature engineering.
2. **Model Training**: Trains both Logistic Regression and Random Forest models on the preprocessed data.
3. **Model Persistence**: Saves the trained models and TF-IDF vectorizer for future predictions.
4. **Evaluation**: Provides accuracy, precision, recall, and F1-score metrics.
5. **Web Application**: A Flask web app that allows users to input an email, detect if it is spam, and generate a suggested response if it is not spam.

## Features

- **Data Loading and Preprocessing**: Loads multiple CSV files, combines them, and extracts features relevant to spam classification.
- **Feature Engineering**: Counts occurrences of common spam words and performs TF-IDF vectorization.
- **Model Training**: Uses Logistic Regression and Random Forest models to classify emails.
- **Model Evaluation**: Calculates and displays performance metrics like accuracy, precision, recall, and F1-score.
- **Model Saving**: Saves the trained models and vectorizer for deployment in a web app.
- **Web Application**: Provides a user interface to input emails, select a model, and get spam detection results along with suggested responses for non-spam emails.

## Dataset

The dataset used is a combination of various sources of labeled emails[^1].
It is stored in five separate CSV files in `./src/Dataset/`. These files are concatenated into a single DataFrame for processing. Each file should contain the following columns:

- `text_combined`: The email text to classify.
- `label`: The label for each email (e.g., `spam` or `not spam`).

## Training the Models

The script `src/model.py`:
1. Loads and concatenates the dataset files using the [`get_dataset`](src/model.py) function.
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

## Web Application

The web application is built using Flask and provides the following features:
- **Email Input**: Users can input the email text they want to classify.
- **Model Selection**: Users can select between Logistic Regression and Random Forest models.
- **Spam Detection**: The app detects if the input email is spam.
- **Suggested Response**: If the email is not spam, the app uses a self-hosted Llama 3.2 model with Ollama to generate a suggested response.

## Setup Instructions

1. **Clone the repository**:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Create a virtual environment**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Download the dataset**:
    Place the dataset CSV files in the `./src/Dataset/` directory.

5. **Train the models**:
    Run the `src/model.py` script to train and save the models.
    ```sh
    python src/model.py
    ```

6. **Set up the Flask web application**:
    ```sh
    export FLASK_APP=src/app.py
    export FLASK_ENV=development  # For development environment
    flask run
    ```

7. **Set up the self-hosted Llama 3.2 with Ollama**:
    Follow the instructions provided by Ollama to set up the self-hosted Llama 3.2 model.

8. **Access the web application**:
    Open your web browser and go to `http://localhost:5000`.

## Future Improvements

1. **Add More Features**: Experiment with additional feature engineering to capture more nuanced aspects of spam emails.
2. **Add More Models**: Test additional classifiers like SVM or Neural Networks.
3. **Hyperparameter Tuning**: Use `GridSearchCV` or `RandomizedSearchCV` for improved model performance.
4. **Optimize Web App**: Develop a user-friendly and responsive front end for the web app, and consider deploying it to a cloud platform like AWS or Heroku.

[^1]: Al-Subaiey, A., Al-Thani, M., Alam, N. A., Antora, K. F., Khandakar, A., & Zaman, S. A. U. (2024, May 19). Novel Interpretable and Robust Web-based AI Platform for Phishing Email Detection. ArXiv.org. https://arxiv.org/abs/2405.11619