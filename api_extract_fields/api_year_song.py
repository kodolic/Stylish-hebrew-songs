import json
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import logging
from concurrent.futures import ThreadPoolExecutor
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_release_year_from_wikipedia(artist_name, song_name):
    search_url = f"https://he.wikipedia.org/wiki/Special:Search?search={artist_name} {song_name}"
    logging.info(f"Searching Wikipedia for {artist_name} {song_name}")
    try:
        response = requests.get(search_url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error accessing Wikipedia: {e}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')
    for result in soup.select('.mw-search-result-heading a'):
        link = "https://he.wikipedia.org" + result['href']
        try:
            page_response = requests.get(link)
            page_response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error accessing Wikipedia page: {e}")
            continue

        page_soup = BeautifulSoup(page_response.text, 'html.parser')
        for paragraph in page_soup.select('p'):
            if song_name in paragraph.text:
                match = re.search(r'\b(19|20)\d{2}\b', paragraph.text)
                if match:
                    logging.info(f"Found release year in Wikipedia: {match.group()}")
                    return match.group()

    return None


HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Cache-Control': 'max-age=0',
    'TE': 'Trailers',
}

def get_release_year_from_shironet_api(url):
    session = requests.Session()
    session.headers.update(HEADERS)
    try:
        response = session.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error accessing Shironet: {e}")
        return None

    if "captcha" in response.text.lower():
        logging.error("Encountered CAPTCHA. Please solve it manually.")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')

    # Locate the span element containing the year
    year_span = soup.find('span', class_='artist_color_gray')
    if year_span:
        year_text = year_span.get_text(strip=True)
        match = re.search(r'\b(19|20)\d{2}\b', year_text)
        if match:
            logging.info(f"Found release year in Shironet: {match.group()}")
            return match.group()

    return None



def get_release_year_from_lastfm_album_page(album_url):
    try:
        response = requests.get(album_url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error accessing Last.fm album page: {e}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')
    try:
        release_date_element = soup.find('dt', text='Release Date').find_next_sibling('dd')
        if release_date_element:
            release_date_text = release_date_element.text.strip()
            match = re.search(r'\b(19|20)\d{2}\b', release_date_text)
            if match:
                logging.info(f"Found release year in Last.fm album page: {match.group()}")
                return match.group()
    except AttributeError:
        logging.error(f"Release date element not found on Last.fm album page: {album_url}")

    return None


def get_release_year_from_lastfm(artist_name, song_name, api_key):
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
            return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error accessing Last.fm: {e}")
        return None

    if 'track' in data and 'album' in data['track']:
        album_url = data['track']['album'].get('url')
        if album_url:
            return get_release_year_from_lastfm_album_page(album_url)

    return None


def get_release_year(artist_name, song_name, api_key, shironet_url):
    try:
        year = get_release_year_from_lastfm(artist_name, song_name, api_key)
        if year:
            return year

        year = get_release_year_from_shironet_api(shironet_url)
        if year:
            return year

        year = get_release_year_from_wikipedia(artist_name, song_name)
        if year:
            return year
    except Exception as e:
        logging.error(f"Error fetching release year for {artist_name} - {song_name}: {e}")

    return None


def fetch_release_year(row, api_key):
    artist_name = row['artist']
    song_name = row['song']
    url = row['url']
    logging.info(f"Fetching release year for {artist_name} - {song_name}")
    return get_release_year(artist_name, song_name, api_key, url)


def update_csv_with_release_year(csv_file_path, api_key, batch_size=100):
    df = pd.read_csv(csv_file_path)

    progress_file_path = csv_file_path.replace('.csv', '_progress.txt')

    # Load the progress from the file if it exists
    start_index = 0
    # if os.path.exists(progress_file_path):
    #     with open(progress_file_path, 'r') as progress_file:
    #         start_index = int(progress_file.read().strip())

    for start in range(start_index, len(df), batch_size):
        end = min(start + batch_size, len(df))
        df_batch = df.iloc[start:end]

        # Only process rows where release_year is null
        rows_to_process = df_batch[df_batch['release_year'].isnull()]

        with ThreadPoolExecutor(max_workers=10) as executor:
            release_years = list(
                executor.map(lambda row: fetch_release_year(row, api_key), [row for _, row in rows_to_process.iterrows()]))

        df.loc[rows_to_process.index, 'release_year'] = release_years

        df.to_csv(csv_file_path, index=False)
        logging.info(f"Updated CSV saved to {csv_file_path}")

        data_for_json = df[['artist', 'song', 'release_year']].to_dict(orient='records')
        json_file_path = csv_file_path.replace('.csv', '_data.json')
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(data_for_json, json_file, ensure_ascii=False, indent=4)
        logging.info(f"JSON file saved to {json_file_path}")
        logging.info(f"Processed rows {start} to {end}")

        # Save the progress
        with open(progress_file_path, 'w') as progress_file:
            progress_file.write(str(end))

# Example usage
api_key = "8c77f14169cbeef3cad090e5de66f50d"
csv_file_path = '../updated_kaggle.csv'
update_csv_with_release_year(csv_file_path, api_key)
