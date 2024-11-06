import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import os
import pickle


# Common spam words indicative of spam
SPAM_WORDS = [
    "free", "win", "winner", "cash", "prize", "bonus", "buy", "cheap", "discount",
    "$$$", "earn", "credit", "investment", "income", "profit", "urgent", "act now",
    "limited time", "don’t miss", "hurry", "offer expires", "last chance",
    "important", "immediately", "fast", "instant", "quick", "now", "final notice",
    "congratulations", "satisfaction guaranteed", "risk-free", "100% free",
    "amazing", "incredible", "once in a lifetime", "no obligation", "free access",
    "special promotion", "click here", "call now", "no fees", "debt", "loan",
    "credit card", "mortgage", "lowest rate", "refinance", "consolidate", "billing",
    "limited offer", "free trial", "check", "pills", "miracle", "cure", 
    "weight loss", "drug", "health", "doctor", "solution", "anti-aging",
    "trial", "order now", "extra cash", "gift", "sample", "quote", "satisfaction"
]

# Directory to save trained models
MODEL_DIR = "./src/trained_models/"


# Since the dataset is too big, it is split into 5 files then concatenated in 1 dataframe
def get_dataset() -> pd.DataFrame:
    files = ["./src/Dataset/phishing_email-1.csv", "./src/Dataset/phishing_email-2.csv", 
             "./src/Dataset/phishing_email-3.csv", "./src/Dataset/phishing_email-4.csv", 
             "./src/Dataset/phishing_email-5.csv"]
    
    datasets = [pd.read_csv(file) for file in files]
    
    final_dataset = pd.concat(datasets, ignore_index=True)
    
    return final_dataset


# Save model or vectorizer to pkl file
def save_model_to_pkl(model, name: str):
    path = os.path.join(MODEL_DIR, f"{name}_model.pkl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        print(f"[+] - Saved the model in {path}")
    except Exception as e:
        print(f"Error saving model: {e}")


# Train the passed model and save it in a pickle file for later use
def train_and_save_model(model, X_train, y_train, X_test, y_test):
    model_name = type(model).__name__

    # Train the model
    start = time.time()
    print(f"[+] - Beginning the training of {model_name} model.")
    
    model.fit(X_train, y_train)
    
    end = time.time()
    print(f"[+] - Finished training the model in {int(end-start)} seconds.")
    
    # Save the trained model in a pickle file
    save_model_to_pkl(model, model_name)
    
    # Evaluate the model's performance
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    
    print("[----------------- Evaluation Metrics -----------------]")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")
        


def main():
    dataset = get_dataset()

    ## Feature Engineering
    # Count the occurence of common spam trigger words in each email
    dataset["spam_word_count"] = dataset["text_combined"].apply(lambda x: sum(x.lower().count(word) for word in SPAM_WORDS))
    # TODO : Add more features ?

    
    # Encoding the label to numerical values
    encoder = LabelEncoder()
    y = encoder.fit_transform(dataset["label"])
    print("[+] - Encoded labels to numerical values.")
    
    
    # Splitting the dataset into a 80/20 train/test split
    X_train_text, X_test_text, X_train_additional, X_test_additional, y_train, y_test = train_test_split(
        dataset["text_combined"], dataset[["spam_word_count"]], y, test_size=0.2, random_state=42
    )


    # Vectorizing the content of the emails into numerical features. Using unigram and bi-grams 
    # with Tf-Idf helps capture the significance of patterns of words like "Click Here"
    # Limit the number of feature to reduce training time. Only include words appearing in 5 different emails
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000, min_df=5)
    
    
    # Fit and transform the training text data. Save the vectorizer to a pkl file
    X_train_tfidf = tfidf.fit_transform(X_train_text)
    save_model_to_pkl(tfidf, "tfidf")

    # Transform the test text data (using the vocabulary and IDF values learned from the training set)
    X_test_tfidf = tfidf.transform(X_test_text)

    # Convert additional features to sparse matrices
    X_train_additional_sparse = csr_matrix(X_train_additional.values)
    X_test_additional_sparse = csr_matrix(X_test_additional.values)

    # Combine TF-IDF and additional features for both training and test sets
    X_train_features = hstack([X_train_tfidf, X_train_additional_sparse])
    X_test_features = hstack([X_test_tfidf, X_test_additional_sparse])
    
    print("[+] - Finished processing the features.")
    
    
    # Train the model, save it, and check its performance
    model2 = RandomForestClassifier(random_state=42)
    model1 = LogisticRegression(random_state=42, max_iter=1000)
    
    train_and_save_model(model1, X_train_features, y_train, X_test_features, y_test)
    train_and_save_model(model2, X_train_features, y_train, X_test_features, y_test)



if __name__ == "__main__":
    main()


