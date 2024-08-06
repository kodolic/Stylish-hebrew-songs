import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import re

import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the main dataset
file_path = 'datasetMappedStyles.csv.csv'  # Ensure this file is in the correct directory
df = pd.read_csv(file_path)

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def get_sentence_embedding(sentence):
    # Tokenize input
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)

    # Get the hidden states from BERT
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the embeddings for the [CLS] token
    cls_embeddings = outputs[0][:, 0, :].numpy()

    return cls_embeddings


def calculate_semantic_similarity(song_lyrics, translated_lyrics):
    if pd.isna(song_lyrics) or pd.isna(translated_lyrics):
        return None

    # Get embeddings for original and translated lyrics
    original_embedding = get_sentence_embedding(song_lyrics)
    translated_embedding = get_sentence_embedding(translated_lyrics)

    # Calculate cosine similarity
    similarity = cosine_similarity(original_embedding, translated_embedding)[0][0]

    return similarity

# Filter the main dataset for the songs with missing values in the semantic_similarity column
df_missing_avg_word_freq = df[df['semantic_similarity'].isna()]

# Add a column for the semantic similarity of the translated song
df_missing_avg_word_freq['semantic_similarity'] = df_missing_avg_word_freq.apply(
    lambda row: calculate_semantic_similarity(row['words'], row['songInEnglish']), axis=1
)

# Update the main dataset with the new column
df.update(df_missing_avg_word_freq[['name', 'artist', 'semantic_similarity']])

# Save the updated dataset with the new column
updated_csv_path = '../datasetMappedStyles.csv'
df.to_csv(updated_csv_path, index=False)

# Print the path to the updated dataset
print(f"Updated dataset saved to: {updated_csv_path}")
