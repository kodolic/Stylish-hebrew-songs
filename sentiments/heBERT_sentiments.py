import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import ast

# Setup logging for debugging purposes
logging.basicConfig(level=logging.INFO)

# Load the dataset
df = pd.read_csv('../datasetMappedStyles_final.csv')

# Check for missing or empty song lyrics and filter them out
df = df.dropna(subset=['words'])

# Convert the 'songs' column from lists of tokens to strings

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("avichr/heBERT_sentiment_analysis")
model = AutoModelForSequenceClassification.from_pretrained("avichr/heBERT_sentiment_analysis")

# Create the sentiment analysis pipeline
sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Function to get sentiment
def get_sentiment(text):
    truncated_text = text[:512]  # Ensure text is within model's input length limits
    result = sentiment_analysis(truncated_text)
    logging.info(f"result: {result}")
    return result

# Function to convert sentiment labels to scores
def sentiment_to_score(sentiment_result):
    try:
        if isinstance(sentiment_result, list) and len(sentiment_result) > 0:
            label = sentiment_result[0]['label']
            score = sentiment_result[0]['score']
            logging.info(f"label: {label}, score: {score}")
            if label == "positive":  # Positive
                return score
            elif label == "negative":  # Negative
                print(sentiment_result)
                return -score
            else:  # Neutral
                return 0
        else:
            logging.error(f"Unexpected sentiment result format: {sentiment_result}")
            return 0
    except Exception as e:
        logging.error(f"Error converting sentiment result to score: {e}")
        return 0

# Apply sentiment analysis and convert to scores
df['heBERT_sentiment'] = df['words'].apply(lambda x: sentiment_to_score(get_sentiment(x)))

# Print min and max sentiment scores
min_sentiment = df['heBERT_sentiment'].min()
max_sentiment = df['heBERT_sentiment'].max()
print(f"Min heBERT sentiment score: {min_sentiment}")
print(f"Max heBERT sentiment score: {max_sentiment}")



# Save the updated DataFrame to the same CSV file
df.to_csv('kaggle_sentiment_23.csv', index=False)

print("The dataset has been updated with heBERT sentiment scores and saved to 'updated_kaggle.csv'.")
