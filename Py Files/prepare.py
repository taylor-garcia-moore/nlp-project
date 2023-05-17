#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import re
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer

def clean_readme_data(df):
    # Convert text to lowercase
    df['Readme'] = df['Readme'].apply(lambda x: " ".join(x.lower() for x in x.split()))

    # Remove HTML tags
    df['Readme'] = df['Readme'].apply(lambda x: BeautifulSoup(x, 'html.parser').get_text())

    # Remove punctuation marks
    df['Readme'] = df['Readme'].str.replace('[^\w\s]', '')

    # Remove numerical digits
    df['Readme'] = df['Readme'].apply(lambda x: re.sub(r'\d+', '', x))

    # Remove URLs
    df['Readme'] = df['Readme'].apply(lambda x: re.sub(r'http\S+|www.\S+', '', x))

    # Remove special characters
    df['Readme'] = df['Readme'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))

    # Remove duplicate words
    df['Readme'] = df['Readme'].apply(lambda x: ' '.join(set(x.split())))

    return df



# In[ ]:


from nltk.tokenize import word_tokenize
from nltk.tokenize import word_tokenize
from nltk.sentiment.util import mark_negation

def preprocess_text(df):
    def expand_contractions(text):
        contraction_mapping = {
            "can't": "cannot",
            "won't": "will not",
            # Add more contractions and their expansions as needed
        }
        words = text.split()
        expanded_words = [contraction_mapping[word.lower()] if word.lower() in contraction_mapping else word for word in words]
        return ' '.join(expanded_words)

    def split_negations(text):
        tokenized_text = word_tokenize(text)
        split_text = mark_negation(tokenized_text)
        return ' '.join(split_text)

    def expand_abbreviations(text):
        abbreviation_mapping = {
            "USA": "United States of America",
            # Add more abbreviations and their expansions as needed
        }
        words = text.split()
        expanded_words = [abbreviation_mapping[word] if word in abbreviation_mapping else word for word in words]
        return ' '.join(expanded_words)

    def remove_short_or_long_words(text, min_length, max_length):
        words = text.split()
        filtered_words = [word for word in words if len(word) >= min_length and len(word) <= max_length]
        return ' '.join(filtered_words)

    df['Readme'] = df['Readme'].apply(expand_contractions)
    df['Readme'] = df['Readme'].apply(split_negations)
    df['Readme'] = df['Readme'].apply(expand_abbreviations)
    df['Readme'] = df['Readme'].apply(lambda x: remove_short_or_long_words(x, min_length=2, max_length=20))

    return df['Readme']


# In[ ]:


import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

def apply_pos_tagging_and_lemmatization(df):
    # Convert 'Readme' column to string
    df['Readme'] = df['Readme'].astype(str)

    # POS Tagging
    def pos_tagging(text):
        tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        return pos_tags

    # Lemmatization
    def lemmatize(pos_tags):
        lemmatizer = WordNetLemmatizer()
        lemmas = []
        for word, pos in pos_tags:
            # Convert POS tags to WordNet POS tags
            wn_pos = get_wordnet_pos(pos)
            if wn_pos:
                lemma = lemmatizer.lemmatize(word, pos=wn_pos)
            else:
                lemma = lemmatizer.lemmatize(word)
            lemmas.append(lemma)
        return lemmas

    # Helper function to convert POS tags to WordNet POS tags
    def get_wordnet_pos(pos_tag):
        if pos_tag.startswith('J'):
            return wordnet.ADJ
        elif pos_tag.startswith('V'):
            return wordnet.VERB
        elif pos_tag.startswith('N'):
            return wordnet.NOUN
        elif pos_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    # Apply POS tagging and lemmatization to the 'Readme' column
    df['POS_tags'] = df['Readme'].apply(pos_tagging)
    df['Lemmas'] = df['POS_tags'].apply(lemmatize)

    return df


# In[ ]:


import nltk
from nltk.corpus import stopwords



from nltk.corpus import stopwords
import re

def apply_stopword_removal(df):
    # Get the default set of stopwords
    default_stopwords = set(stopwords.words('english'))

    # Add prepositions to the stopwords set
    prepositions = ['about', 'above', 'across', 'after', 'against', 'along', 'among', 'around',
                    'as', 'at', 'before', 'behind', 'below', 'beneath', 'beside', 'between',
                    'beyond', 'but', 'by', 'concerning', 'considering', 'despite', 'down',
                    'during', 'except', 'for', 'from', 'in', 'inside', 'into', 'like',
                    'near', 'of', 'off', 'on', 'onto', 'out', 'outside', 'over', 'past',
                    'regarding', 'round', 'since', 'through', 'throughout', 'to', 'toward',
                    'under', 'underneath', 'until', 'unto', 'up', 'upon', 'with', 'within',
                    'without', 'http', 'https', 'com'] 

    stopwords_with_prepositions = default_stopwords.union(prepositions)

    # Apply stopwords removal to 'Readme' column
    stop = set(stopwords_with_prepositions)
    df['Readme'] = df['Readme'].apply(lambda x: " ".join(word for word in re.findall(r'\b\w+\b', x.lower()) if word not in stop))

    return df
from nltk.corpus import stopwords
import re

def apply_stopwords_removal(df):
    # Get the default set of stopwords
    default_stopwords = set(stopwords.words('english'))

    # Add prepositions to the stopwords set
    prepositions = ['about', 'above', 'across', 'after', 'against', 'along', 'among', 'around',
                    'as', 'at', 'before', 'behind', 'below', 'beneath', 'beside', 'between',
                    'beyond', 'but', 'by', 'concerning', 'considering', 'despite', 'down',
                    'during', 'except', 'for', 'from', 'in', 'inside', 'into', 'like',
                    'near', 'of', 'off', 'on', 'onto', 'out', 'outside', 'over', 'past',
                    'regarding', 'round', 'since', 'through', 'throughout', 'to', 'toward',
                    'under', 'underneath', 'until', 'unto', 'up', 'upon', 'with', 'within',
                    'without', 'http', 'https', 'com']

    stopwords_with_prepositions = default_stopwords.union(prepositions)

    # Apply stopwords removal to 'Readme' column
    stop = set(stopwords_with_prepositions)
    df['Readme'] = df['Readme'].apply(lambda x: " ".join(word for word in x if word not in stop))  # Removed the unnecessary x.split()

    return df

# In[ ]:
from collections import Counter

def extract_common_words(df):
    # Combine all the Readme text into a single string
    all_text = ' '.join(df['Readme'])

    # Tokenize the text into words
    words = all_text.split()

    # Count the frequency of each word
    word_counts = Counter(words)

    # Extract the most common words
    common_words = word_counts.most_common(10)

    return common_words


from sklearn.feature_extraction.text import CountVectorizer

def extract_ngrams(df):
    
    df['Readme'] = df['Readme'].apply(nltk.word_tokenize)
    # Convert the 'Readme' column into a string
    readme_text = [' '.join(words) for words in df['Readme']]

    # Create a CountVectorizer instance for bigrams
    bigram_vectorizer = CountVectorizer(ngram_range=(2, 2))

    # Fit and transform the 'Readme' column for bigrams
    X_bigram = bigram_vectorizer.fit_transform(readme_text)

    # Get the feature names (bigrams)
    bigram_features = bigram_vectorizer.get_feature_names()

    # Create a CountVectorizer instance for trigrams
    trigram_vectorizer = CountVectorizer(ngram_range=(3, 3))

    # Fit and transform the 'Readme' column for trigrams
    X_trigram = trigram_vectorizer.fit_transform(readme_text)

    # Get the feature names (trigrams)
    trigram_features = trigram_vectorizer.get_feature_names()

    # Print the bigrams and trigrams
    print("Bigrams:")
    print(bigram_features)
    print("\nTrigrams:")
    print(trigram_features)
    
from sklearn.feature_extraction.text import CountVectorizer

def extracted_ngrams(df):
    df['Readme'] = df['Readme'].apply(nltk.word_tokenize)
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

    if len(bigram_freqs) == 0:
        print("No bigrams available. Documents may contain only stop words.")
    else:
        # Print the top 10 most frequent bigrams
        top_bigrams = Counter(bigram_freqs).most_common(10)
        print("Top 10 Bigrams:")
        for bigram, count in top_bigrams:
            print(f"{bigram}: {count}")
