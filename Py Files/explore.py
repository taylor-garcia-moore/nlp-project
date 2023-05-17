#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import base64
import re
from collections import Counter
from bs4 import BeautifulSoup
common_words = []

import pandas as pd
from collections import Counter
import re

def analyze_readme_data(df):
    # Create a dictionary to categorize words by programming language
    word_categories = {word: [] for word in df['Language'].unique()}

    # Create a dictionary to store the length of READMEs by language
    length_by_language = {}

    # Loop through each repository in the DataFrame
    for repo, readme_text, language in zip(df['Repository'], df['Readme'], df['Language']):
        # Extract words from the README text
        words = re.findall(r"\b\w+\b", readme_text.lower())

        # Calculate the length of the README
        length = len(readme_text)

        # Update length by language dictionary
        if language in length_by_language:
            length_by_language[language].append(length)
        else:
            length_by_language[language] = [length]

        # Categorize words by programming language
        for word in words:
            if word in word_categories:
                word_categories[word].append(repo)

    # Find the most common words in READMEs
    common_words = Counter(words).most_common(10)

    # Calculate the average length of READMEs by programming language
    average_length_by_language = {
        language: sum(lengths) / len(lengths)
        for language, lengths in length_by_language.items()
    }

    # Create a DataFrame with the results
    result_df = pd.DataFrame({
        'Language': list(average_length_by_language.keys()),
        'Average README Length': list(average_length_by_language.values())
    })

    return result_df, common_words, average_length_by_language

import pandas as pd
import explore

def merge_analyze_results(df):
    result_df, common_words, average_length_by_language = explore.analyze_readme_data(df)

    # Convert common words to DataFrame
    common_words_df = pd.DataFrame(common_words, columns=['Word', 'Count'])

    # Convert average length by language to DataFrame
    average_length_by_language_df = pd.DataFrame(average_length_by_language.items(), columns=['Language', 'Average_Readme_Length'])

    # Merge the dataframes
    merged_df = pd.merge(result_df, common_words_df, left_index=True, right_index=True)
    merged_df = pd.merge(merged_df, average_length_by_language_df, on='Language', how='left')

    # Merge merged_df with the original df DataFrame
    df = pd.merge(df, merged_df, on='Language', how='left')
    # removes "Average README Length" column
    df.drop("Average README Length", axis=1, inplace=True)
    # Return the updated DataFrame
    return df



# In[ ]:
common_words =[]

def plot_common_words(common_words):
    common_words = Counter(words).most_common(10)
    # Plot the most common words
    words = [word for word, count in common_words]
    counts = [count for word, count in common_words]

    plt.barh(words, counts)
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    plt.title('Most Common Words in READMEs')
    plt.show()


# In[ ]:


import matplotlib.pyplot as plt

def plot_average_length_by_language(languages, lengths):
    # Filter out None values from languages and lengths
    languages_filtered = [lang for lang in languages if lang is not None]
    lengths_filtered = [length for lang, length in zip(languages, lengths) if lang is not None]

    # Plot the average README length by language
    plt.plot(languages_filtered, lengths_filtered, marker='o')
    plt.xlabel('Programming Language')
    plt.ylabel('Average README Length')
    plt.title('Average README Length by Language')
    plt.xticks(rotation=90)
    plt.show()


# In[ ]:


import matplotlib.pyplot as plt


import numpy as np

def plot_average_length_by_language_barh(languages, lengths):
    # Filter out None values and NaN values from languages and lengths
    data = [(lang, length) for lang, length in zip(languages, lengths) if lang is not None and not np.isnan(length)]

    # Extract filtered languages and lengths
    languages_filtered = [lang for lang, length in data]
    lengths_filtered = [int(length) for lang, length in data]

    # Plot the average README length by language using a horizontal bar chart
    plt.barh(languages_filtered, lengths_filtered)
    plt.xlabel('Average README Length')
    plt.ylabel('Programming Language')
    plt.title('Average README Length by Language')
    plt.show()




# In[ ]:

import re
from collections import Counter
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from collections import Counter

import re
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from collections import Counter

import re
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from collections import Counter

def visualize_top_word(df):
    # Extract words from the README text
    words = [re.findall(r"\b\w+\b", readme_text.lower()) for readme_text in df['Readme']]
    words = [word for sublist in words for word in sublist]

    # Add prepositions to the stopwords set
    prepositions = ['about', 'above', 'across', 'after', 'against', 'along', 'among', 'around',
                    'as', 'at', 'before', 'behind', 'below', 'beneath', 'beside', 'between',
                    'beyond', 'but', 'by', 'concerning', 'considering', 'despite', 'down',
                    'during', 'except', 'for', 'from', 'in', 'inside', 'into', 'like',
                    'near', 'of', 'off', 'on', 'onto', 'out', 'outside', 'over', 'past',
                    'regarding', 'round', 'since', 'through', 'throughout', 'to', 'toward',
                    'under', 'underneath', 'until', 'unto', 'up', 'upon', 'with', 'within',
                    'without','htttp','https','com']

    stopwords_set = set(stopwords.words('english')).union(prepositions)

    # Exclude stopwords from the words list
    words = [word for word in words if word not in stopwords_set]

    # Find the most common words in READMEs
    common_words = Counter(words).most_common(10)

    # Extract the words and counts for plotting
    words = [word for word, count in common_words]
    counts = [count for word, count in common_words]

    # Plot the most common words
    plt.barh(words, counts)
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    plt.title('Top 10 Common Words in READMEs')
    plt.show()



def visualize_top_words(df):
    # Extract words from the README text
    words = [re.findall(r"\b\w+\b", readme_text.lower()) for readme_text in df['Readme']]
    words = [word for sublist in words for word in sublist]

    # Find the most common words in READMEs
    common_words = Counter(words).most_common(10)

    # Extract the words and counts for plotting
    words = [word for word, count in common_words]
    counts = [count for word, count in common_words]

    # Plot the most common words
    plt.barh(words, counts)
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    plt.title('Top 10 Common Words in READMEs')
    plt.show()
import pandas as pd



def process_dataframes(df):
    top_languages = ['Python', 'C++', 'JavaScript']

    # Loop through the DataFrame rows
    for index, row in df.iterrows():
        # Check if the language is in the top languages list
        if row['Language'] in top_languages:
            # Assign the language label to the 'Language' column
            df.at[index, 'Language'] = row['Language']
        else:
            # Assign 'Other' label to the 'Language' column
            df.at[index, 'Language'] = 'Other'

    return df

from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import matplotlib.pyplot as plt

def visualize_top_bigrams(df):
    # Convert the 'Readme' column into a string
    readme_text = [' '.join(words) for words in df['Readme']]

    # Create a CountVectorizer instance for bigrams
    bigram_vectorizer = CountVectorizer(ngram_range=(2, 2))
    X_bigram = bigram_vectorizer.fit_transform(readme_text)

    # Get the feature names (bigrams)
    bigram_features = bigram_vectorizer.get_feature_names()

    # Calculate the frequencies of bigrams
    bigram_counts = X_bigram.sum(axis=0).A1
    bigram_freqs = dict(zip(bigram_features, bigram_counts))

    # Get the top 10 most frequent bigrams
    top_bigrams = Counter(bigram_freqs).most_common(10)

    # Prepare data for visualization
    top_bigrams = top_bigrams[::-1]  # Reverse the order for horizontal bar chart
    bigrams = [bigram for bigram, count in top_bigrams]
    counts = [count for bigram, count in top_bigrams]

    # Visualize the top 10 bigrams
    plt.barh(bigrams, counts)
    plt.xlabel('Frequency')
    plt.ylabel('Bigrams')
    plt.title('Top 10 Bigrams')
    plt.show()
    
    
import pandas as pd
import matplotlib.pyplot as plt

def plot_language_distribution(df, column):
    # Function to return value counts as a dataframe
    def value_counts_df(df, column):
        absolute = pd.DataFrame(df[column].value_counts())
        percent = pd.DataFrame(df[column].value_counts(normalize=True))
        df_value_counts = pd.concat([absolute, percent], axis=1)
        df_value_counts.columns = ['n', 'percent']
        return df_value_counts

    # Apply value_counts_df to the specified column
    df_value_counts = value_counts_df(df, column)

    # Plot the value counts as a horizontal bar chart
    plt.figure(figsize=(10, 6))
    df_value_counts['n'].plot(kind='barh')
    plt.xlabel('Count')
    plt.ylabel(column)
    plt.title(f'{column} Distribution')
    plt.tight_layout()
    plt.show()
    
    
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

def get_top_bigrams(df, n=10):
    # Convert the 'Readme' column into a string
    readme_text = [' '.join(words) for words in df['Readme']]

    # Create a CountVectorizer instance for bigrams
    bigram_vectorizer = CountVectorizer(ngram_range=(2, 2))
    X_bigram = bigram_vectorizer.fit_transform(readme_text)

    # Get the feature names (bigrams)
    bigram_features = bigram_vectorizer.get_feature_names()

    # Calculate the frequencies of bigrams
    bigram_counts = X_bigram.sum(axis=0).A1
    bigram_freqs = dict(zip(bigram_features, bigram_counts))

    # Get the top n most frequent bigrams
    top_bigrams = Counter(bigram_freqs).most_common(n)
    
    
    # Print the top 10 most frequent bigrams
    print("Top 10 Bigrams:")
    for bigram, count in top_bigrams:
        print(f"{bigram}: {count}")
    return top_bigrams


def add_word_count_and_average_length(df):
    # Function to calculate the word count
    def calculate_word_count(text):
        tokens = text.split()  # Split the text into individual words
        word_count = len(tokens)  # Count the number of words
        return word_count

    # Function to calculate the average word length
    def calculate_average_word_length(text):
        tokens = text.split()  # Split the text into individual words
        total_length = sum(len(word) for word in tokens)  # Sum the lengths of all words
        word_count = len(tokens)  # Count the number of words
        average_length = total_length / word_count  # Calculate the average word length
        return average_length

    # Create new columns for word count and average word length
    df['Word_Count'] = df['Readme'].apply(calculate_word_count)
    df['Average_Word_Length'] = df['Readme'].apply(calculate_average_word_length)


