import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Load the dataset
file_path = '../datasetMappedStyles.csv'  # Update with the correct file path
data = pd.read_csv(file_path)

# Descriptive Statistics Calculation
descriptive_stats = data.describe(include='all')
print("Descriptive Statistics:")
print(descriptive_stats)

# Correlation Analysis
correlation_matrix = data.corr(numeric_only=True)
print("Correlation Matrix:")
print(correlation_matrix)

# Regression Analysis
# Define the independent variables and the dependent variable
independent_vars = data[['wordCount', 'uniqueWords', 'releaseYear', 'numberOfRepeatedWords', 'ratioOfTotalWordsToUnique', 'percentageOfTotalWordsToUnique', 'DiffLemmas', 'DiffPOS', 'numberOfBiGrams', 'numberOfTriGrams', 'bigramsEntropy', 'trigramsEntropy', 'averageSetWordLength', 'WordsRhymes', 'RatioOfPOStoWords', 'readabilityMeasure', 'positiveWords', 'negativeWords', 'avgSimilarityMeasure']]
dependent_var = data['sentimentScore']

# Add a constant to the independent variables
independent_vars = sm.add_constant(independent_vars)

# Perform the regression analysis
model = sm.OLS(dependent_var, independent_vars).fit()

# Display the summary of the regression analysis
print("Regression Analysis Summary:")
print(model.summary())

# Comparative Analysis
# Group data by Music Style and calculate mean values for each group
music_style_comparison = data.groupby('Music Style').mean(numeric_only=True)
print("Music Style Comparison:")
print(music_style_comparison)

# Visualization
# Visualization of sentiment score distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['sentimentScore'], kde=True, bins=30)
plt.title('Distribution of Sentiment Scores')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.show()

# Visualization of correlation heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Visualization of average sentiment score by music style
plt.figure(figsize=(10, 6))
sns.barplot(x=music_style_comparison.index, y=music_style_comparison['sentimentScore'])
plt.title('Average Sentiment Score by Music Style')
plt.xlabel('Music Style')
plt.ylabel('Average Sentiment Score')
plt.xticks(rotation=45)
plt.show()
