import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import re

# Function to clean and standardize music style labels
def clean_and_standardize_music_style(style):
    if pd.isna(style) or not style.strip():
        return None
    style = str(style)
    # Remove any character that is not a letter or space
    cleaned_style = re.sub(r'[^a-zA-Z\s,]', '', style)
    # Remove unwanted characters after specific words like 'Hip-Hop'
    cleaned_style = re.sub(r'Hip-Hop.*', 'Hip-Hop', cleaned_style)
    # Split by commas, trim whitespace, remove empty genres, sort genres, and join them back
    genres = [genre.strip() for genre in cleaned_style.split(',') if genre.strip()]
    if not genres:
        return None
    genres.sort()
    return ', '.join(genres)

# Load the dataset
df = pd.read_csv('../datasetMappedStyles.csv')

# Select features that describe creativity
creativity_features = [
    'uniqueWords', 'ratioOfTotalWordsToUnique', 'percentageOfTotalWordsToUnique',
    'DiffLemmas', 'DiffPOS', 'bigramsEntropy', 'trigramsEntropy', 'sentimentScore',
    'averageSetWordLength', 'WordsRhymes', 'RatioOfPOStoWords', 'readabilityMeasure',
    'avgSimilarityMeasure',  'average_word_frequency','avg_word_similarity_hebrew','avg_word_similarity_english'
]

# Invert the similarity and frequency features to align with the creativity score
df['inv_avgSimilarityMeasure'] = 1 - df['avgSimilarityMeasure']
df['inv_word_similarity'] = 1 - df['word_similarity-large']
df['inv_average_word_frequency'] = 1 - df['average_word_frequency']
df['inv_avg_word_similarity_hebrew'] = 1 - df['avg_word_similarity_hebrew']
df['inv_avg_word_similarity_english'] = 1 - df['avg_word_similarity_english']

# List of features after inverting
adjusted_creativity_features = [
    'uniqueWords', 'ratioOfTotalWordsToUnique', 'percentageOfTotalWordsToUnique',
    'DiffLemmas', 'DiffPOS', 'bigramsEntropy', 'trigramsEntropy',
    'averageSetWordLength', 'WordsRhymes', 'RatioOfPOStoWords','NumberOfUniqueWordsby1/freq',
    'inv_avgSimilarityMeasure', 'inv_average_word_frequency','inv_avg_word_similarity_hebrew','inv_avg_word_similarity_english'
]

# Apply cleaning and standardization
df['Music Style'] = df['Music Style'].apply(clean_and_standardize_music_style)

# Drop rows where Music Style is None
df = df.dropna(subset=['Music Style'])

# Normalize each creativity feature
scaler = MinMaxScaler()
df[adjusted_creativity_features] = scaler.fit_transform(df[adjusted_creativity_features])

# Combine features into a single creativity score
df['creativity_score'] = df[adjusted_creativity_features].mean(axis=1)
creativity_over_years = df.groupby('releaseYear')['creativity_score'].mean().reset_index()

# Apply rolling average to smooth the values over the years
creativity_over_years['creativity_score_smooth'] = creativity_over_years['creativity_score'].rolling(window=5, center=True).mean()

# Plot creativity score over the years
plt.figure(figsize=(15, 8))
sns.lineplot(data=creativity_over_years, x='releaseYear', y='creativity_score_smooth')
plt.title('Creativity Score Over the Years (Smoothed)')
plt.xlabel('Release Year')
plt.ylabel('Average Creativity Score')
plt.tight_layout()
plt.show()

# Aggregate creativity score by music style
creativity_by_style = df.groupby('Music Style')['creativity_score'].mean().reset_index()

# Plot creativity score by music style
plt.figure(figsize=(15, 8))
sns.barplot(data=creativity_by_style, x='Music Style', y='creativity_score')
plt.title('Creativity Score by Music Style')
plt.xlabel('Music Style')
plt.ylabel('Average Creativity Score')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Analyze Specific Features Over the Years
features_over_years = df.groupby('releaseYear')[adjusted_creativity_features].mean().reset_index()

# Apply rolling average to smooth the values over the years for each feature
for feature in adjusted_creativity_features:
    features_over_years[feature + '_smooth'] = features_over_years[feature].rolling(window=5, center=True).mean()

# Plot each feature over the years
plt.figure(figsize=(15, 10))
for feature in adjusted_creativity_features:
    sns.lineplot(data=features_over_years, x='releaseYear', y=feature + '_smooth', label=feature)

plt.title('Creativity Features Over the Years (Smoothed)')
plt.xlabel('Release Year')
plt.ylabel('Average Feature Value')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# Aggregate creativity score by music style
creativity_by_style = df.groupby('Music Style')['creativity_score'].mean().reset_index()

# Plot creativity score by music style
plt.figure(figsize=(15, 8))
sns.barplot(data=creativity_by_style, x='Music Style', y='creativity_score')
plt.title('Creativity Score by Music Style')
plt.xlabel('Music Style')
plt.ylabel('Average Creativity Score')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Analyze Specific Features Over the Years
features_over_years = df.groupby('releaseYear')[adjusted_creativity_features].mean().reset_index()

# Plot each feature over the years
plt.figure(figsize=(15, 10))
for feature in adjusted_creativity_features:
    sns.lineplot(data=features_over_years, x='releaseYear', y=feature, label=feature )

plt.title('Creativity Features Over the Years')
plt.xlabel('Release Year')
plt.ylabel('Average Feature Value')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Analyze Specific Features Over Music Styles
features_over_music_style = df.groupby('Music Style')[adjusted_creativity_features].mean().reset_index()

# Plot each feature over music styles
plt.figure(figsize=(15, 10))
for feature in adjusted_creativity_features:
    sns.lineplot(data=features_over_music_style, x='Music Style', y=feature, label=feature)

plt.title('Creativity Features Over Music Styles')
plt.xlabel('Music Style')
plt.ylabel('Average Feature Value')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
