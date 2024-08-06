import logging
import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import json
from concurrent.futures import ThreadPoolExecutor

def get_genres_from_lastfm_album_page(album_url):
    try:
        response = requests.get(album_url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error accessing Last.fm album page: {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    genres = []
    try:
        tags_container = soup.find_all('a', href=True)
        genres = [tag.get_text() for tag in tags_container if '/tag/' in tag['href']]
        logging.info(f"Found genres in Last.fm album page: {genres}")
    except AttributeError:
        logging.error(f"Genres not found on Last.fm album page: {album_url}")

    return genres

def get_genres_from_lastfm(artist_name, song_name, api_key):
    url = "http://ws.audioscrobbler.com/2.0/"
    params = {
        "method": "track.getInfo",
        "api_key": api_key,
        "artist": artist_name,
        "track": song_name,
        "format": "json"
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        try:
            data = response.json()
        except json.decoder.JSONDecodeError:
            logging.error("Invalid JSON response from Last.fm")
            return []
    except requests.exceptions.RequestException as e:
        logging.error(f"Error accessing Last.fm: {e}")
        return []

    if 'track' in data and 'album' in data['track']:
        album_url = data['track']['album'].get('url')
        if album_url:
            return get_genres_from_lastfm_album_page(album_url)

    return []

def fetch_genres(row, api_key):
    artist_name = row['artist']
    song_name = row['song']
    url = row['url']
    logging.info(f"Fetching genres for {artist_name} - {song_name}")
    return get_genres_from_lastfm(artist_name, song_name, api_key)

def update_csv_with_genres(csv_file_path, api_key, batch_size=100):
    df = pd.read_csv(csv_file_path)

    # Add the 'genres' column if it doesn't exist
    if 'genres' not in df.columns:
        df['genres'] = ""

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
            genres_list = list(
                executor.map(lambda row: fetch_genres(row, api_key), [row for _, row in rows_to_process.iterrows()]))

        df.loc[rows_to_process.index, 'genres'] = ['; '.join(genres) for genres in genres_list]

        df.to_csv(csv_file_path, index=False)
        logging.info(f"Updated CSV saved to {csv_file_path}")

        data_for_json = df[['artist', 'song', 'genres']].to_dict(orient='records')
        json_file_path = csv_file_path.replace('.csv', '_data.json')
        with open('genere_per_song', 'w', encoding='utf-8') as json_file:
            json.dump(data_for_json, json_file, ensure_ascii=False, indent=4)
        logging.info(f"JSON file saved to {json_file_path}")
        logging.info(f"Processed rows {start} to {end}")

        # Save the progress
        with open(progress_file_path, 'w') as progress_file:
            progress_file.write(str(end))

api_key = "8c77f14169cbeef3cad090e5de66f50d"
csv_file_path = '../updated_kaggle.csv'
update_csv_with_genres(csv_file_path, api_key)
