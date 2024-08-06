import json
import logging
import os
import time
import requests
import spotipy
import yaml
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def load_config(path="config.yaml"):
    with open(path, 'r') as file:
        cfg = yaml.safe_load(file)
    return cfg

config = load_config()

client_credentials_manager = SpotifyClientCredentials(client_id=config['spotify']['client_id'],
                                                      client_secret=config['spotify']['client_secret'])

sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def get_genres_by_track(artist_name, track_name):
    try:
        results = sp.search(q=f'track:{track_name} artist:{artist_name}', type='track')
        if results['tracks']['items']:
            track = results['tracks']['items'][0]
            album = sp.album(track['album']['id'])
            return album['genres']
        else:
            return []
    except Exception as e:
        logging.error(f"Error fetching genres for {artist_name} - {track_name}: {e}")
        return []

def fetch_genres(row):
    artist_name = row['artist']
    track_name = row['song']
    logging.info(f"Fetching genres for {artist_name} - {track_name}")
    return get_genres_by_track(artist_name, track_name)

def update_csv_with_genres(csv_file_path, batch_size=100, delay=5):
    df = pd.read_csv(csv_file_path)

    if 'genres' not in df.columns:
        df['genres'] = None

    progress_file_path = csv_file_path.replace('.csv', '_progress.txt')

    # Load the progress from the file if it exists
    start_index = 0
    if os.path.exists(progress_file_path):
        with open(progress_file_path, 'r') as progress_file:
            start_index = int(progress_file.read().strip())

    for start in range(start_index, len(df), batch_size):
        end = min(start + batch_size, len(df))
        df_batch = df.iloc[start:end]

        # Only process rows where genres is null or empty
        rows_to_process = df_batch[df_batch['genres'].isnull() | (df_batch['genres'] == '')]

        with ThreadPoolExecutor(max_workers=10) as executor:
            genres_list = list(executor.map(fetch_genres, [row for _, row in rows_to_process.iterrows()]))

        df.loc[rows_to_process.index, 'genres'] = ['; '.join(genres) for genres in genres_list]

        df.to_csv(csv_file_path, index=False)
        logging.info(f"Updated CSV saved to {csv_file_path}")

        data_for_json = df[['artist', 'song', 'genres']].to_dict(orient='records')
        json_file_path = csv_file_path.replace('.csv', '_data.json')
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(data_for_json, json_file, ensure_ascii=False, indent=4)
        logging.info(f"JSON file saved to {json_file_path}")
        logging.info(f"Processed rows {start} to {end}")

        # Save the progress
        with open(progress_file_path, 'w') as progress_file:
            progress_file.write(str(end))

        # Delay between batches to avoid rate limits
        time.sleep(delay)

# Example usage
if __name__ == "__main__":
    update_csv_with_genres("../updated_kaggle.csv")
