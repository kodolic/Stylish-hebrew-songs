import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the main dataset
file_path = 'updated_with_avg_word_similarity_both.csv'  # Ensure this file is in the correct directory
df = pd.read_csv(file_path)

# Load pre-trained multilingual BERT model and tokenizer for Hebrew
tokenizer_multilingual = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model_multilingual = BertModel.from_pretrained('bert-base-multilingual-cased')

# Load pre-trained BERT model and tokenizer for English
tokenizer_english = BertTokenizer.from_pretrained('bert-base-uncased')
model_english = BertModel.from_pretrained('bert-base-uncased')

def get_word_embeddings(sentence, tokenizer, model):
    # Tokenize input
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)

    # Get the hidden states from BERT
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the embeddings for each word (excluding special tokens)
    word_embeddings = outputs[0].squeeze(0)[1:-1].numpy()  # Remove [CLS] and [SEP] tokens

    return word_embeddings

def calculate_average_similarity(song_lyrics, tokenizer, model):
    if pd.isna(song_lyrics):
        return None

    # Get embeddings for lyrics
    word_embeddings = get_word_embeddings(song_lyrics, tokenizer, model)

    # Calculate cosine similarity for each pair of word embeddings
    similarities = cosine_similarity(word_embeddings)

    # Get the upper triangular matrix, excluding the diagonal
    upper_tri_indices = np.triu_indices_from(similarities, k=1)
    upper_tri_similarities = similarities[upper_tri_indices]

    # Calculate the average similarity
    avg_similarity = np.mean(upper_tri_similarities) if len(upper_tri_similarities) > 0 else None

    return avg_similarity

# Filter rows with missing values for avg_word_similarity_hebrew
missing_hebrew = df['avg_word_similarity_hebrew'].isna()

# Calculate and update avg_word_similarity_hebrew for missing values
df.loc[missing_hebrew, 'avg_word_similarity_hebrew'] = df.loc[missing_hebrew, 'words'].apply(
    lambda x: calculate_average_similarity(x, tokenizer_multilingual, model_multilingual))

# Filter rows with missing values for avg_word_similarity_english
missing_english = df['avg_word_similarity_english'].isna()

# Calculate and update avg_word_similarity_english for missing values
df.loc[missing_english, 'avg_word_similarity_english'] = df.loc[missing_english, 'translatedWords'].apply(
    lambda x: calculate_average_similarity(x, tokenizer_english, model_english))

# Save the updated dataset with the new columns
updated_csv_path = '../datasetMappedStyles.csv'
df.to_csv(updated_csv_path, index=False)

# Print the path to the updated dataset
print(f"Updated dataset saved to: {updated_csv_path}")
