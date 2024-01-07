# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import networkx as nx
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import pandas as pd
from itertools import combinations, product
import time
from memory_profiler import memory_usage

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load Reddit data
try:
    reddit_data = pd.read_csv("reddit_posts.csv")
    reddit_data.columns = ['Post']
    print(reddit_data.head())
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Function to transform text to sequences
def transform_text_to_sequences(dataframe, column_name):
    stop_words = set(stopwords.words('english'))
    sequences = []

    for post in dataframe[column_name]:
        # Tokenize the post
        tokens = word_tokenize(post)
        # Remove stopwords and non-alphabetic words
        filtered_tokens = [word.lower() for word in tokens if word.isalpha() and word not in stop_words]
        sequences.append(filtered_tokens)

    return sequences

# Function to measure GSP algorithm efficiency
def measure_gsp_efficiency(sequences, min_support, max_gap, max_length):
    start_time = time.time()
    mem_usage_before = memory_usage()

    # Run the GSP algorithm
    frequent_items = gsp_algorithm(sequences, min_support, max_gap, max_length)

    mem_usage_after = memory_usage()
    end_time = time.time()

    elapsed_time = end_time - start_time
    memory_used = max(mem_usage_after) - max(mem_usage_before)

    print(f"Time taken: {elapsed_time} seconds")
    print(f"Memory used: {memory_used} MiB")

    return frequent_items

# Apply the transformation to the Reddit dataset
reddit_sequences = transform_text_to_sequences(reddit_data, 'Post')

# Function to check if candidate is within max gap in a sequence
def is_within_max_gap(sequence, candidate, max_gap):
    indices = [i for item in candidate for i, seq_item in enumerate(sequence) if item == seq_item]
    for combination in combinations(indices, 2):
        if abs(combination[0] - combination[1]) > max_gap:
            return False
    return True

# Function to count support considering the max gap
def count_support_with_gap(sequences, candidates, max_gap):
    support_counts = {candidate: 0 for candidate in candidates}
    for sequence in sequences:
        for candidate in candidates:
            if candidate.issubset(sequence) and is_within_max_gap(sequence, candidate, max_gap):
                support_counts[candidate] += 1
    return support_counts

# Efficient candidate generation with join and prune steps
def join_step(frequent_items, length):
    candidates = set()
    items = list(frequent_items)
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            combined_set = items[i].union(items[j])
            if len(combined_set) == length:
                candidates.add(frozenset(combined_set))
    return candidates

def prune_step(candidates, previous_frequent, length):
    return {candidate for candidate in candidates if all(frozenset(combination) in previous_frequent for combination in combinations(candidate, length - 1))}

# GSP algorithm with maxGap and short rules
def gsp_algorithm(sequences, min_support, max_gap, max_length):
    length = 2
    frequent_sequences = set()
    current_frequent_items = {frozenset([item]) for sequence in sequences for item in sequence}

    while length <= max_length:
        candidates = join_step(current_frequent_items, length)
        candidates = prune_step(candidates, current_frequent_items, length)
        support_counts = count_support_with_gap(sequences, candidates, max_gap)
        current_frequent = {candidate for candidate, count in support_counts.items() if count >= min_support}

        if not current_frequent:
            break

        frequent_sequences.update(current_frequent)
        current_frequent_items = current_frequent
        length += 1

    return frequent_sequences

def get_frequent_items_with_counts(frequent_items, sequences):
    item_counts = {}
    for item_set in frequent_items:
        for sequence in sequences:
            if item_set.issubset(set(sequence)):
                item_counts[item_set] = item_counts.get(item_set, 0) + 1
    return list(item_counts.items())

# Visualization function for frequent items
def visualize_frequent_items(frequent_items_with_counts, max_sets_to_plot=5):
    # Sort and limit the number of item sets to visualize
    frequent_items_with_counts = sorted(frequent_items_with_counts, key=lambda x: x[1], reverse=True)[:max_sets_to_plot]

    # Create separate graphs for each frequent item set
    num_sets = len(frequent_items_with_counts)
    fig, axes = plt.subplots(nrows=num_sets, ncols=2, figsize=(15, num_sets * 5))
    if num_sets == 1:
        axes = [axes]

    for i, (item_set, count) in enumerate(frequent_items_with_counts):
        # Network graph
        G = nx.Graph()
        item_list = list(item_set)
        G.add_nodes_from(item_list)
        G.add_edges_from(combinations(item_list, 2))

        # Node size based on count of appearances
        node_size = [100 for _ in item_list]

        # Draw the graph for this item set
        pos = nx.spring_layout(G, k=0.15, iterations=20)
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='skyblue', alpha=0.7, ax=axes[i, 0])
        nx.draw_networkx_edges(G, pos, alpha=0.2, ax=axes[i, 0])
        nx.draw_networkx_labels(G, pos, font_size=8, ax=axes[i, 0])

        axes[i, 0].set_title(f"Frequent Item Set: {' & '.join(item_set)} (Count: {count})", fontsize=16)
        axes[i, 0].axis('off')

        # Bar graph
        items, freqs = zip(*[(item, freq) for item in item_set for freq in [count]])
        axes[i, 1].bar(items, freqs, color='orange')
        axes[i, 1].set_title(f"Frequency of Items in Set: {' & '.join(item_set)}")
        axes[i, 1].set_ylabel('Frequency')
        axes[i, 1].set_xticklabels(items, rotation=45)

    plt.tight_layout()
    plt.show()


# Example usage of the algorithm
min_support = 10
max_gap = 3
max_length = 3  # For short rules
frequent_items = measure_gsp_efficiency(reddit_sequences, min_support, max_gap, max_length)

# Convert frequent items to format suitable for visualization
frequent_items_with_counts = get_frequent_items_with_counts(frequent_items, reddit_sequences)

# Visualize the frequent items with their counts
visualize_frequent_items(frequent_items_with_counts)
