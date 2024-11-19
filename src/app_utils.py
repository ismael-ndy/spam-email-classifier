import requests
import json
import pickle
import string
import os
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from model import MODEL_DIR, SPAM_WORDS


# Prompt passed to the LLM to generate a response to the email
def get_pompt(email_text: str) -> str:
    prompt = f"""
    You are an AI email assistant. Your task is to draft a professional and concise email response based on the provided email input. Use polite, formal language unless otherwise specified. Always ensure clarity and a friendly tone.

    Input Email: {email_text}

    Guidelines:

    Address the recipient properly, using "Dear [Name]" or a similar polite greeting.
    Answer any specific questions or address key points from the input email.
    If a request is made, acknowledge it and provide relevant next steps or details.
    Include an appropriate closing line such as "Best regards" or "Looking forward to your reply."
    Ensure proper grammar and coherence throughout the email.
    I want you to only return the suggested email body. Don't use phrases like : "Here's a draft of a professional and concise email response"
    """
    
    return prompt

# Make a request to the selfhosted Ollama server running llama3.2 to generate a response to the non spam email
def gen_response(email_text: str) -> str:
    prompt = get_pompt(email_text)

    data = {
        "model": "llama3.2:1b",
        "prompt": prompt,
        "stream": False
    }
    url = "http://localhost:11434/api/generate"
    r = requests.post(url=url, json=data)
    
    response = json.loads(r.text)
    
    return response["response"]


# De-pickle the model pickle file enabling us to use it
def load_model(model_name: str):
    '''
    model_name: Options include LogisticRegression, RandomForestClassifier, tfidf.
    '''
    try:
        path = os.path.join(MODEL_DIR, f"{model_name}_model.pkl")
        model = pickle.load(open(path,'rb'))
    except Exception as e:
        print(f"Error loading model: {e}")

    return model


# Predict wether the email provided is spam based on the model selected.
def predict_spam(model_name, processed_email) -> bool:
    model = load_model(model_name)

    predictions = model.predict(processed_email)

    return predictions[0] == 1


# Process the input email to use it in the models
def process_email(email):
    # Remove punctuation and set email to lowercase
    pre_processed_email = email.translate(str.maketrans('', '',
                                    string.punctuation)).lower()
    
    # Count the number of occurences of spam words in the email, puts it in a matrix
    spam_word_count = sum(pre_processed_email.count(word) for word in SPAM_WORDS)
    spam_word_count_matrix = csr_matrix(spam_word_count)

    # Load model to transform text to vectors
    tfidf_model = load_model("tfidf")
    # Convert text to list as TF-IDF needs a list as input
    vectorized_email = tfidf_model.transform([pre_processed_email])

    # Combine text and word count feature into one matrix
    processed_email = hstack([vectorized_email, spam_word_count_matrix])

    return processed_email


# Main method that classify the input email
def spam_classification(email, model_name):
    # Vectorize the input email
    processed_email = process_email(email)

    # Predict wether the email is spam or ham
    is_spam = predict_spam(model_name=model_name, processed_email=processed_email)


    # If the email is not spam, provide a suggested response
    if is_spam:
        return "spam"
    else:
        suggested_response = gen_response(email)
        return suggested_response



    


    