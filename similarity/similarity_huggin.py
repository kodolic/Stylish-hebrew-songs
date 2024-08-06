import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load the pre-trained multilingual model and tokenizer
model_name = "intfloat/multilingual-e5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def embed_words(words):
    inputs = tokenizer(words, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs[0][:, 0, :].cpu()  # Use the embeddings of the [CLS] token and move to CPU
    return embeddings

def average_word_similarity(words):
    if not words:  # Check for empty or missing lyrics
        return np.nan
    embeddings = embed_words(words)
    num_words = embeddings.shape[0]
    similarities = []

    for i in range(num_words):
        for j in range(i + 1, num_words):
            embedding1 = embeddings[i].reshape(1, -1)
            embedding2 = embeddings[j].reshape(1, -1)
            similarity = cosine_similarity(embedding1, embedding2)
            similarities.append(similarity[0][0])

    return np.mean(similarities) if similarities else np.nan

# Load the dataset
file_path = '../datasetMappedStyles.csv'  # Update this path
df = pd.read_csv(file_path)

# Determine the batch size
batch_size = 100  # Adjust this size based on your system's memory capacity

# Check for last processed song
last_processed_path = '../sentiments/last_processed_song.txt'
if os.path.exists(last_processed_path):
    with open(last_processed_path, 'r') as f:
        last_processed_index = int(f.read().strip())
else:
    last_processed_index = 0

# Find rows with missing 'word_similarity-large' values
df_missing = df[df['word_similarity-large'].isna()]
print(df_missing)
# Process the dataset in batches
for start in range(last_processed_index, len(df_missing), batch_size):
    end = min(start + batch_size, len(df_missing))
    batch = df_missing.iloc[start:end]

    # Apply the similarity calculation to each song's lyrics in the batch
    batch['word_similarity-large'] = batch['words'].apply(lambda x: average_word_similarity(x.split()))

    # Update the main dataset with the new values
    df.loc[batch.index, 'word_similarity-large'] = batch['word_similarity-large']

    # Save the intermediate results
    df.to_csv('datasetMappedStyles_final.csv', index=False, encoding='utf-8-sig')

    # Update the last processed song
    with open(last_processed_path, 'w') as f:
        f.write(str(end))

    print(f"Processed songs from {start} to {end}")

print("Updated dataset with word similarity scores saved successfully.")
