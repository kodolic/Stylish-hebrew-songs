import pandas as pd
import re
from wordfreq import word_frequency

# Load the main dataset
file_path = 'updated_with_avg_word_similarity_both.csv'  # Ensure this file is in the correct directory
df = pd.read_csv(file_path)

# Remove duplicate rows
df = df.drop_duplicates()

def calculate_average_word_frequency(song_lyrics):
    if pd.isna(song_lyrics):
        return 0.0
    words = re.findall(r'\b\w+\b', song_lyrics)
    if not words:
        return 0.0
    total_frequency = sum(word_frequency(word, 'he', wordlist='large') for word in words)
    average_frequency = total_frequency / len(words)
    return average_frequency

# Filter the main dataset for the songs with missing values in the average_word_frequency column
df_missing_avg_word_freq = df[df['average_word_frequency'].isna()]

# Add a column for the average word frequency of each song
df_missing_avg_word_freq['average_word_frequency'] = df_missing_avg_word_freq['words'].apply(calculate_average_word_frequency)

# Update the main dataset with the new column
df.update(df_missing_avg_word_freq[['name', 'artist', 'average_word_frequency']])

# Save the updated dataset with the new column
updated_csv_path = 'datasetMappedStyles.csv.csv'
df.to_csv(updated_csv_path, index=False)

# Print the path to the updated dataset
print(f"Updated dataset saved to: {updated_csv_path}")
