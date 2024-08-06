import pandas as pd
from arabic_reshaper import reshape
from bidi import get_display
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import json
import matplotlib.pyplot as plt

def load_and_prepare_data(filename, column, values):
    # Load your dataset
    df = pd.read_csv(filename)

    # Filter the dataset for the specified values (artists or genres)
    filtered_df = df[df[column].isin(values)]

    # Find the minimum number of songs across the specified values
    min_count = filtered_df[column].value_counts().min()
    print(f"Minimum count across specified {column}s: {min_count}")

    # Sample min_count songs from each value
    sampled_df = filtered_df.groupby(column).apply(lambda x: x.sample(min_count)).reset_index(drop=True)
    return sampled_df

def train_and_evaluate_model(df, column, features):
    X = df[features]
    y = df[column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)

    # Train a RandomForestClassifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Predict on the test set
    y_pred = clf.predict(X_test)

    # Calculate accuracy for each value (artist or genre)
    values = df[column].unique()
    results = {}
    for value in values:
        mask = (y_test == value)
        value_accuracy = accuracy_score(y_test[mask], y_pred[mask])
        results[value] = {'Accuracy': value_accuracy, 'Predicted': list(y_pred[mask]), 'Actual': list(y_test[mask])}

    # Feature importances
    importances = clf.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    sorted_feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    return results, sorted_feature_importance_df

def save_results(results, feature_importances, results_filename, importance_filename):
    # Save the results to a JSON file
    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=4)

    # Save the feature importances to a JSON file
    with open(importance_filename, 'w') as f:
        json.dump(feature_importances.to_dict(orient='records'), f, indent=4)


def get_artist_styles(df):
    # Create a dictionary mapping artists to their music styles
    artist_styles = df.groupby('artist')['Music Style'].apply(lambda x: ', '.join(x.unique())).to_dict()
    return artist_styles


def visualize_results(results, feature_importances, column, df=None):
    # Visualize the results
    # Plot the accuracies

    fig, ax = plt.subplots(figsize=(10, 6))
    values = list(results.keys())
    accuracies = [results[value]['Accuracy'] for value in values]

    if column == 'artist' and df is not None:
        artist_styles = get_artist_styles(df)
        values = [f"({get_display(reshape(artist_styles[value]))} {get_display(reshape(value))} )" for value in values]

    ax.bar(values, accuracies, color='skyblue', capsize=5)
    ax.set_title('Classification Accuracies')
    ax.set_xlabel(column)
    ax.set_ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.xticks(rotation=0, ha='center')  # Rotate x-axis labels for better readability
    plt.show()

    # Plot feature importances
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(feature_importances['Feature'], feature_importances['Importance'], color='lightgreen')
    ax.set_title('Feature Importances')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    plt.gca().invert_yaxis()
    plt.show()

def print_results(results, feature_importances):
    # Print the sorted feature importances
    print("Feature Importances (from most to least important):")
    for index, row in feature_importances.iterrows():
        print(f"{row['Feature']}: {row['Importance']:.4f}")

    # Print the accuracies
    print("\nGenre/Artist Accuracies:")
    for value, result in results.items():
        print(f"{value} Accuracy: {result['Accuracy']:.2f}")

def print_highest_values(df, column, features):
    grouped_means = df.groupby(column)[features].mean()
    higher_values = grouped_means.idxmax()
    print("Feature with highest value for each artist/genre:")
    for feature, value in higher_values.items():
        print(f"{feature}: {value}")
def main(filename, column, values, features, results_filename, importance_filename):
    df = load_and_prepare_data(filename, column, values)
    results, feature_importances = train_and_evaluate_model(df, column, features)
    save_results(results, feature_importances, results_filename, importance_filename)
    print_results(results, feature_importances)
    print_highest_values(df,column,features)
    visualize_results(results, feature_importances,column,df)


if __name__ == "__main__":
    filename = '../datasetMappedStyles.csv'
    column = 'Music Style'
    values = ["Pop,Rock", "Mizrahi,Pop"]
    # column = 'artist'
    # values = ["גלי עטרי", "ירדנה ארזי"]
    features = ['wordCount', 'uniqueWords', 'numberOfRepeatedWords', 'ratioOfTotalWordsToUnique',
                'percentageOfTotalWordsToUnique', 'DiffLemmas', 'DiffPOS', 'numberOfBiGrams', 'numberOfTriGrams',
                'bigramsEntropy', 'trigramsEntropy', 'sentimentScore', 'averageSetWordLength', 'WordsRhymes',
                'RatioOfPOStoWords', 'readabilityMeasure', 'positiveWords', 'negativeWords', 'avgSimilarityMeasure',
                'NumberOfUniqueWordsby1/freq', 'AvgUniqueness', 'percentageOfRepeatedWords',
                'theUniquenessLvlOfTheRepeatedSongs', 'semantic_similarity', 'average_word_frequency', 'heBERT_sentiment',
                'avg_word_similarity_hebrew', 'avg_word_similarity_english', 'word_similarity-large']
    results_filename = 'classification_results.json'
    importance_filename = 'feature_importances.json'

    main(filename, column, values, features, results_filename, importance_filename)
