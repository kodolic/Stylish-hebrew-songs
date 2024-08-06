import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer, AutoModel
import torch
import json

# Load song data from CSV
df = pd.read_csv('../datasetMappedStyles.csv')

# Function to split music styles and handle missing values
def split_music_styles(styles):
    if pd.isna(styles):
        return []
    return styles.split(', ') if ', ' in styles else [styles]

# Apply the function to split music styles into a list of styles
df['Music Styles'] = df['Music Style'].apply(split_music_styles)

# Filter for specific genres and ensure 3,000 songs per genre in training and 1,000 songs per genre in testing
genres = ['Mizrahi', 'Rock', 'Pop']
df_filtered = df[df['Music Styles'].apply(lambda x: any(genre in x for genre in genres))]

# Function to sample a fixed number of songs per genre
def sample_songs_per_genre(df, genre, train_size, test_size):
    genre_df = df[df['Music Styles'].apply(lambda x: genre in x)]
    train_sample = genre_df.sample(n=min(train_size, len(genre_df)), random_state=42)
    remaining = genre_df.drop(train_sample.index)
    test_sample = remaining.sample(n=min(test_size, len(remaining)), random_state=42)
    return train_sample, test_sample

# Sample songs for each genre
train_samples = []
test_samples = []
for genre in genres:
    train_sample, test_sample = sample_songs_per_genre(df_filtered, genre, 3000, 1000)
    train_samples.append(train_sample)
    test_samples.append(test_sample)

# Concatenate all training and testing samples
train_df = pd.concat(train_samples).drop_duplicates(subset=['artist', 'words'])
test_df = pd.concat(test_samples).drop_duplicates(subset=['artist', 'words'])

# Initialize DeBERTa tokenizer and model
try:
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base', use_fast=False)
    model = AutoModel.from_pretrained('microsoft/deberta-v3-base')
except ValueError as e:
    print("Error initializing tokenizer or model:", e)
    print("Make sure sentencepiece is installed and try again.")
    import sys
    sys.exit(1)

def embed_text(text):
    """Function to get the embedding of a given text using DeBERTa."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Preprocess and get embeddings for each song
def get_embeddings(df):
    lyrics = df['words'].apply(eval)
    return np.array([embed_text(" ".join(lyrics)) for lyrics in lyrics])

train_embeddings = get_embeddings(train_df)
test_embeddings = get_embeddings(test_df)

# Hierarchical clustering with a fixed number of clusters
n_clusters = 3  # Adjust this value as needed
cluster = AgglomerativeClustering(n_clusters=n_clusters).fit(train_embeddings)

# Compute centroids of clusters
def compute_centroids(embeddings, labels):
    return np.array([embeddings[labels == i].mean(axis=0) for i in range(n_clusters)])

# Initialize the music styles dictionary
music_styles_dict = {genre: {"count": 0, "predict_success": 0} for genre in genres}
artists_dict = {}

def add_to_dict(key, dictionary):
    if key not in dictionary:
        dictionary[key] = {"count": 1, "predict_success": 0}
    else:
        print(f"{key} already exists.")

def update_dict(real_values, predicted_value, dictionary):
    real_values = real_values.split(', ')
    for real_value in real_values:
        if real_value in dictionary:
            dictionary[real_value]["count"] += 1
            if predicted_value == real_value:
                dictionary[real_value]["predict_success"] += 1

def print_sorted_dict(dictionary):
    # Calculate success rate for each music style
    success_rates = {
        key: (data["predict_success"] / data["count"] * 100) if data["count"] > 0 else 0
        for key, data in dictionary.items()
    }

    # Sort the music styles by success rate in decreasing order
    sorted_dict = sorted(success_rates.items(), key=lambda item: item[1], reverse=True)

    # Print the sorted music styles and their success rates
    for key, success_rate in sorted_dict:
        print(f"{key}: {success_rate:.2f}% success rate")

# Predict artist based on song lyrics using k-NN within the cluster
def predict(lyrics, k_music_style=30, k_artist=11):
    # Embed lyrics
    embedded_lyrics = embed_text(" ".join(lyrics))
    # Convert embeddings and centroids to torch tensors
    embedded_lyrics_tensor = torch.tensor(embedded_lyrics).unsqueeze(0)
    centroids = compute_centroids(train_embeddings, cluster.labels_)
    centroids_tensor = torch.tensor(centroids)
    # Compute distances to centroids
    distances = torch.cdist(embedded_lyrics_tensor, centroids_tensor, p=2)
    # Find the nearest centroid
    cluster_id = torch.argmin(distances).item()
    print("Predicted Cluster:", cluster_id)
    # Get k-nearest neighbors within the cluster
    cluster_embeddings = train_embeddings[cluster.labels_ == cluster_id]

    # Predict music style
    cluster_music_styles = train_df.iloc[cluster.labels_ == cluster_id]['Music Style']
    neigh_music_style = NearestNeighbors(n_neighbors=min(k_music_style, len(cluster_music_styles)))
    neigh_music_style.fit(cluster_embeddings)
    distances_music_style, indices_music_style = neigh_music_style.kneighbors(embedded_lyrics_tensor.detach().numpy())
    nearest_music_styles = cluster_music_styles.iloc[indices_music_style[0]]
    predicted_music_style = nearest_music_styles.mode().iloc[0]
    print("Nearest Music Style:", nearest_music_styles)

    # Predict artist
    cluster_artists = train_df.iloc[cluster.labels_ == cluster_id]['artist']
    neigh_artist = NearestNeighbors(n_neighbors=min(k_artist, len(cluster_artists)))
    neigh_artist.fit(cluster_embeddings)
    distances_artist, indices_artist = neigh_artist.kneighbors(embedded_lyrics_tensor.detach().numpy())
    nearest_artists = cluster_artists.iloc[indices_artist[0]]
    predicted_artist = nearest_artists.mode().iloc[0]
    print("Nearest Artists:", nearest_artists)

    return predicted_music_style, predicted_artist

# Predicting artists for the prediction dataset
predicted_artists = []
real_artists = []
predicted_music_styles = []
real_music_styles = []

for i in range(len(test_df)):
    real_artist = test_df.iloc[i]['artist']
    real_music_style = test_df.iloc[i]['Music Style']
    new_song_lyrics = eval(test_df.iloc[i]['words'])
    predicted_music_style, predicted_artist = predict(new_song_lyrics)
    real_artists.append(real_artist)
    predicted_artists.append(predicted_artist)
    real_music_styles.append(real_music_style)
    predicted_music_styles.append(predicted_music_style)
    update_dict(real_music_style, predicted_music_style, music_styles_dict)
    update_dict(real_artist, predicted_artist, artists_dict)
    print(f"Real Artist: {real_artist}, Predicted Artist: {predicted_artist}")
    print(f"Real Music Style: {real_music_style}, Predicted Music Style: {predicted_music_style}")

# Save predictions to JSON
predictions = {
    "real_artists": real_artists,
    "predicted_artists": predicted_artists,
    "real_music_styles": real_music_styles,
    "predicted_music_styles": predicted_music_styles,
}

with open('../predictions.json', 'w', encoding='utf-8') as f:
    json.dump(predictions, f, ensure_ascii=False, indent=4)

print("Predictions have been saved to 'predictions.json'.")

# Evaluate performance
from sklearn.metrics import confusion_matrix, accuracy_score

def evaluate_performance(real_values, predicted_values, entity_name):
    cm = confusion_matrix(real_values, predicted_values)
    accuracy = accuracy_score(real_values, predicted_values)
    print(f"Confusion Matrix for {entity_name}:\n", cm)
    print(f"Accuracy for {entity_name}:", accuracy)
    success_percentage = accuracy * 100
    print(f"Percentage of Successful Predictions in {entity_name}:", success_percentage)
    return cm, accuracy, success_percentage

evaluate_performance(real_artists, predicted_artists, "artists")
evaluate_performance(real_music_styles, predicted_music_styles, "music styles")

print_sorted_dict(artists_dict)
print_sorted_dict(music_styles_dict)
