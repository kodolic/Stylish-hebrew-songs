import pandas as pd
import numpy as np
from scipy.stats import entropy
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('../datasetMappedStyles.csv')

# Preprocess data
df['bigramsEntropy'] = pd.to_numeric(df['bigramsEntropy'], errors='coerce')
df['trigramsEntropy'] = pd.to_numeric(df['trigramsEntropy'], errors='coerce')

# Calculate POS Diversity Index (assuming POS tags are separated by spaces)
df['POS_diversity'] = df['POSperWord'].apply(lambda x: len(set(x.split())) / len(x.split()) if x else np.nan)

# Define the genres of interest
genres = ['Mizrahi', 'Rock', 'Pop', 'Hip-Hop']

# Filter the DataFrame to include only rows where 'Music Style' matches one of the genres of interest
filtered_df = df[df['Music Style'].isin(genres)]

# Group by Music Style and calculate average measures of complexity

from sklearn.preprocessing import MinMaxScaler

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Assuming 'filtered_df' contains your filtered data with music styles of interest
# Reshape data for scaling
bigrams_reshaped = filtered_df['bigramsEntropy'].values.reshape(-1, 1)
trigrams_reshaped = filtered_df['trigramsEntropy'].values.reshape(-1, 1)

# Scale the entropy values
filtered_df['bigramsEntropy_normalized'] = scaler.fit_transform(bigrams_reshaped)
filtered_df['trigramsEntropy_normalized'] = scaler.fit_transform(trigrams_reshaped)

# Check the updated DataFrame
print(filtered_df[['Music Style', 'bigramsEntropy_normalized', 'trigramsEntropy_normalized']])
complexity_measures = filtered_df.groupby('Music Style').agg({
    'POS_diversity': 'mean',
    'bigramsEntropy_normalized': 'mean',
    'trigramsEntropy_normalized': 'mean',
    'RatioOfPOStoWords': 'mean'
}).reset_index()
# Print the complexity measures
print(complexity_measures.values)

# Data setup from your complexity_measures DataFrame
music_styles = complexity_measures['Music Style']
pos_diversity = complexity_measures['POS_diversity']
bigrams_entropy_normalized = complexity_measures['bigramsEntropy_normalized']
trigrams_entropy_normalized = complexity_measures['trigramsEntropy_normalized']
ratio_pos_to_words = complexity_measures['RatioOfPOStoWords']

# Bar width
bar_width = 0.2

# X position of groups
r1 = np.arange(len(pos_diversity))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
r4 = [x + bar_width for x in r3]

# Create bars
plt.figure(figsize=(14, 8))
plt.bar(r1, pos_diversity, color='b', width=bar_width, edgecolor='grey', label='POS Diversity')
plt.bar(r2, bigrams_entropy_normalized, color='r', width=bar_width, edgecolor='grey', label='Normalized Bigrams Entropy')
plt.bar(r3, trigrams_entropy_normalized, color='g', width=bar_width, edgecolor='grey', label='Normalized Trigrams Entropy')
plt.bar(r4, ratio_pos_to_words, color='purple', width=bar_width, edgecolor='grey', label='Ratio of POS to Words')

# Add xticks on the middle of the group bars
plt.xlabel('Music Style', fontweight='bold')
plt.ylabel('Metrics', fontweight='bold')
plt.xticks([r + bar_width for r in range(len(pos_diversity))], music_styles, rotation=90)

# Create legend & Show graphic
plt.legend()
plt.title('Linguistic Complexity Measures by Music Style')
plt.tight_layout()
plt.show()