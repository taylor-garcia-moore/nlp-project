#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import prepare, model, explore
from sklearn.metrics import classification_report

def train_and_evaluate_model(df):
    # Drop rows with missing values in 'Language' column
    df = df.dropna(subset=['Language'])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df['Readme'], df['Language'], test_size=0.2, random_state=42)

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Fit the vectorizer on the training data
    X_train_vectors = vectorizer.fit_transform(X_train)

    # Transform the testing data using the fitted vectorizer
    X_test_vectors = vectorizer.transform(X_test)

    # Remove rows with 'None' values from training data
    non_null_indices = y_train.notnull()
    X_train_vectors = X_train_vectors[non_null_indices]
    y_train = y_train[non_null_indices]

    # Train a Support Vector Machine (SVM) classifier
    classifier = SVC()
    classifier.fit(X_train_vectors, y_train)

    # Make predictions on the testing data
    y_pred = classifier.predict(X_test_vectors)

    # Print the classification report with zero_division=1
    print(classification_report(y_test, y_pred, zero_division=1))


# In[ ]:

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_and_evaluate_models(df):
    # Clean the README data
    df['Readme'] = df['Readme'].astype(str).apply(lambda x: " ".join(x.lower() for x in x.split()))

    # Preprocessing and feature extraction
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['Readme'])
    y = df['Language']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate the SVM model
    svm_classifier = SVC()
    svm_classifier.fit(X_train, y_train)
    svm_predictions = svm_classifier.predict(X_test)

    # Print evaluation scores for SVM
    print("SVM Classification Report:")
    print(classification_report(y_test, svm_predictions))

    # Train and evaluate the Naive Bayes model
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train, y_train)
    nb_predictions = nb_classifier.predict(X_test)

    # Print evaluation scores for Naive Bayes
    print("Naive Bayes Classification Report:")
    print(classification_report(y_test, nb_predictions))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report

def evaluate_model(df):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df['Readme'], df['Language'], test_size=0.2, random_state=42)

    X_train = X_train.tolist()  # Convert the Series to a list of strings
    X_test = X_test.tolist()  # Convert the Series to a list of strings

    # Vectorize the tokenized texts
    vectorizer = TfidfVectorizer()
    X_train_vectors = vectorizer.fit_transform(X_train)
    X_test_vectors = vectorizer.transform(X_test)

    # Train a classification model
    svm = SVC()
    svm.fit(X_train_vectors, y_train)

    # Evaluate the model
    y_pred = svm.predict(X_test_vectors)
    accuracy = svm.score(X_test_vectors, y_test)
    print("Model Accuracy:", accuracy)

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def evaluate_models(df):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df['Readme'], df['Language'], test_size=0.2, random_state=42)

    X_train = X_train.tolist()  # Convert the Series to a list of strings
    X_test = X_test.tolist()  # Convert the Series to a list of strings

    # Vectorize the tokenized texts
    vectorizer = TfidfVectorizer()
    X_train_vectors = vectorizer.fit_transform(X_train)
    X_test_vectors = vectorizer.transform(X_test)

    # Train a classification model
    svm = SVC()
    svm.fit(X_train_vectors, y_train)

    # Evaluate the model
    y_pred = svm.predict(X_test_vectors)

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Create a scatter plot of actual vs predicted values
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.show()
