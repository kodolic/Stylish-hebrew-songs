import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('../updated_datasetMappedStyles_final.csv')

# Normalize features to be between -1 and 1
def normalize(series):
    return 2 * ((series - series.min()) / (series.max() - series.min())) - 1

df['positiveWords_norm'] = normalize(df['positiveWords'])
df['negativeWords_norm'] = normalize(df['negativeWords'])
df['sentimentScore_norm'] = normalize(df['sentimentScore'])

# Combine the normalized sentiment features to create a single sentiment score
df['combined_sentiment'] = df[['heBERT_sentiment', 'positiveWords_norm', 'negativeWords_norm', 'sentimentScore_norm']].mean(axis=1)

# Group similar music styles and filter out styles with few data points
style_counts = df['Music Style'].value_counts()
common_styles = style_counts[style_counts > 1000].index  # Keep styles with more than 20 songs

df_filtered = df[df['Music Style'].isin(common_styles)]

# Plotting the sentiment distribution by music style
plt.figure(figsize=(14, 8))

sns.boxplot(x='Music Style', y='combined_sentiment', data=df_filtered)
plt.xlabel('Music Style')
plt.ylabel('Combined Sentiment Score (Normalized)')
plt.title('Sentiment Analysis by Music Style')
plt.xticks(rotation=90)
plt.show()
