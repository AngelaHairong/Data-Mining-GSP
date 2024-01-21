# -*- coding: utf-8 -*-
"""MSF.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rfAYqy9fHATx14NxctAqTZxj4iPd2tcC
"""

pip install scikit-learn
pip install memory_profiler

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from itertools import combinations
import time
from memory_profiler import memory_usage
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer


# Load dataset
data = pd.read_csv("/content/reddit_movies20k.csv", sep = '~')
data.columns = ['posts']

# Basic text preprocessing
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    # Tokenize, remove stopwords, and lemmatize
    tokens = nltk.word_tokenize(text.lower())
    return ' '.join([lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words])

data['processed_posts'] = data['posts'].apply(preprocess)


# Extract keywords
vectorizer = TfidfVectorizer(max_features=50)  # Adjust number of features
tfidf_matrix = vectorizer.fit_transform(data['processed_posts'])
features = vectorizer.get_feature_names_out()  # Updated method name

# Group titles by keywords
keywords_to_titles = {word: [] for word in features}
for post, processed in zip(data['posts'], data['processed_posts']):
    for word in features:
        if word in processed:
            keywords_to_titles[word].append(post)

# Create transactions
transactions = list(keywords_to_titles.values())

raw_transactions = [list(set(transaction)) for transaction in transactions]


# Function to measure GSP algorithm efficiency
def measure_gsp_efficiency(sequences, min_support, max_gap, max_length):
    start_time = time.time()
    mem_usage_before = memory_usage()

    # Run the GSP algorithm
    frequent_items = gsp_algorithm_with_MGen(sequences, min_support, max_gap, max_length)

    mem_usage_after = memory_usage()
    end_time = time.time()

    elapsed_time = end_time - start_time
    memory_used = max(mem_usage_after) - max(mem_usage_before)

    print(f"Time taken: {elapsed_time} seconds")
    print(f"Memory used: {memory_used} MiB")

    return frequent_items

# Function to sample the data for initial frequent sequence estimation
'''def sample_data_for_estimation(dataframe, sample_size):
    sample_df = dataframe.sample(n=sample_size)
    return transform_text_to_sequences(sample_df, 'Post')
'''
# Modified MGen function for candidate sequence generation
def MGen(frequent_sequences, length):
    candidates = set()
    for seq in frequent_sequences:
        if len(seq) == length:
            for additional_seq in frequent_sequences:
                if len(additional_seq) == 1:
                    # Create a new candidate by adding the additional sequence
                    new_candidate = seq | additional_seq
                    # Ensure that the new candidate's length is exactly length + 1
                    if len(new_candidate) == length + 1:
                        candidates.add(frozenset(new_candidate))
    return candidates


# Function to check if candidate is within max gap in a sequence
def is_within_max_gap(sequence, candidate, max_gap):
    indices = [i for item in candidate for i, seq_item in enumerate(sequence) if item == seq_item]
    for combination in combinations(indices, len(candidate)):
        if max(combination) - min(combination) <= max_gap:
            return True
    return False

# Function to count support considering the max gap
def count_support_with_gap(sequences, candidates, max_gap):
    support_counts = {candidate: 0 for candidate in candidates}
    for sequence in sequences:
        for candidate in candidates:
            if candidate.issubset(sequence) and is_within_max_gap(sequence, candidate, max_gap):
                support_counts[candidate] += 1
    return support_counts

# Modified GSP algorithm with MGen integration
def gsp_algorithm_with_MGen(sequences, min_support, max_gap, max_length):
    length = 1
    frequent_sequences = set()
    current_frequent_items = {frozenset([item]) for sequence in sequences for item in sequence}

    while length < max_length:
        length += 1
        candidates = MGen(current_frequent_items, length - 1)
        support_counts = count_support_with_gap(sequences, candidates, max_gap)
        current_frequent = {candidate for candidate, count in support_counts.items() if count >= min_support}

        if not current_frequent:
            break

        frequent_sequences.update(current_frequent)
        current_frequent_items = current_frequent

    return frequent_sequences


def get_frequent_items_with_counts(frequent_items, sequences):
    item_counts = {}
    for item_set in frequent_items:
        for sequence in sequences:
            if item_set.issubset(set(sequence)):
                item_counts[item_set] = item_counts.get(item_set, 0) + 1
    return list(item_counts.items())

# Function to get and print common itemsets with their support count
def print_common_itemsets(frequent_items, sequences):
    item_counts = {}
    for item_set in frequent_items:
        for sequence in sequences:
            if item_set.issubset(set(sequence)):
                item_counts[item_set] = item_counts.get(item_set, 0) + 1

    # Sort item sets by their counts
    sorted_item_counts = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)

    # Print item sets with their counts
    for item_set, count in sorted_item_counts:
        print(f"Item Set: {set(item_set)}, Count: {count}")


# Example usage of the algorithm
min_support = 3
max_gap = 8
max_length = 4 # For short rules
frequent_items = measure_gsp_efficiency(raw_transactions, min_support, max_gap, max_length)

# Convert frequent items to format suitable for visualization
frequent_items_with_counts = get_frequent_items_with_counts(frequent_items, raw_transactions)

# Print common itemsets and their support counts
print_common_itemsets(frequent_items, raw_transactions)
