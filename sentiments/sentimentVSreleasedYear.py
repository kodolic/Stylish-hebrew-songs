import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('../datasetMappedStyles.csv')

# Check if all releaseYear values are numeric and convert them
df['releaseYear'] = pd.to_numeric(df['releaseYear'], errors='coerce')

# Drop rows with NaN releaseYear values
df = df.dropna(subset=['releaseYear'])

# Ensure releaseYear is treated as an integer
df['releaseYear'] = df['releaseYear'].astype(int)

# Normalize features to be between -1 and 1
def normalize(series):
    return 2 * ((series - series.min()) / (series.max() - series.min())) - 1

df['positiveWords_norm'] = normalize(df['positiveWords'])
df['negativeWords_norm'] = normalize(df['negativeWords'])
df['sentimentScore_norm'] = normalize(df['sentimentScore'])

# Combine the normalized sentiment features to create a single sentiment score
df['combined_sentiment'] = df[['heBERT_sentiment', 'positiveWords_norm', 'negativeWords_norm', 'sentimentScore_norm']].mean(axis=1)

# Ensure only numeric columns are included in the groupby operation
numeric_columns = df.select_dtypes(include=[np.number]).columns
df_grouped = df[numeric_columns].groupby(df['releaseYear']).mean()

# Applying a rolling window of 5 years and calculating the mean for the combined sentiment
df_grouped['combined_sentiment_rolling'] = df_grouped['combined_sentiment'].rolling(window=5, center=True).mean()

# Identifying peak years with significant fluctuations
threshold = 0.005
fluctuations = df_grouped['combined_sentiment_rolling'].diff().abs()
peak_years = df_grouped.index[fluctuations > threshold]

# Plotting the combined sentiment over the years with rolling averages
plt.figure(figsize=(14, 8))

plt.plot(df_grouped.index, df_grouped['combined_sentiment_rolling'], label='Combined Sentiment (Adaptive Rolling Avg)', color='cyan')

# Annotate peak years
for year in peak_years:
    plt.annotate(str(year), xy=(year, df_grouped.loc[year, 'combined_sentiment_rolling']),
                 xytext=(year, df_grouped.loc[year, 'combined_sentiment_rolling'] + 0.003),
                 arrowprops=dict(facecolor='black',  shrink=0.05, width=0.5, headwidth=5, headlength=5))

plt.xlabel('Release Year')
plt.ylabel('Sentiment Score (Normalized)')
plt.title('Combined Sentiment Analysis Over the Years (Adaptive Rolling Average)')
plt.legend()
plt.show()
