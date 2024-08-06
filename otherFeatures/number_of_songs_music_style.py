import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from bidi.algorithm import get_display
from arabic_reshaper import reshape
# Load the dataset
df = pd.read_csv('../datasetMappedStyles.csv')

# Function to add count annotations to bar plots
def add_annotations(ax, values):
    for idx, value in enumerate(values):
        ax.annotate(f'{int(value)}',
                    (idx, value),
                    ha='center', va='bottom', xytext=(0, 10), textcoords='offset points')

# Set a font that supports Hebrew characters
plt.rcParams['font.family'] = 'DejaVu Sans'  # You can change this to another font that supports Hebrew

# Plot number of songs by music style
def plot_songs_by_style(df):
    song_count_by_style = df['Music Style'].value_counts().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    ax = song_count_by_style.plot(kind='bar', color='skyblue')
    plt.title('Number of Songs by Music Style')
    plt.xlabel('Music Style')
    plt.ylabel('Number of Songs')
    plt.xticks(rotation=45, ha='right')
    add_annotations(ax, song_count_by_style)
    plt.tight_layout()
    plt.show()

# Plot number of songs by artist (top 30)
def plot_songs_by_artist(df):
    top_artists = df['artist'].value_counts().head(30).sort_values(ascending=False)
    # Create a dictionary of artist to music style
    artist_styles = df.groupby('artist')['Music Style'].apply(lambda x: ', '.join(set(x))).to_dict()

    # Combine artist names with their music styles
    reshaped_top_artists = [f"{get_display(reshape(artist))} ({get_display(reshape(artist_styles[artist]))})"
                            for artist in top_artists.index]

    plt.figure(figsize=(12, 6))
    ax = top_artists.plot(kind='bar', color='lightgreen')
    plt.title('Number of Songs by Top 30 Artists')
    plt.xlabel('Artist')
    plt.ylabel('Number of Songs')
    ax.set_xticklabels(reshaped_top_artists, rotation=45, ha='right')

    add_annotations(ax, top_artists)
    plt.tight_layout()
    plt.show()



# Plot number of songs by decade
def plot_songs_by_decade(df,label):
    df['Decade'] = (df[label] // 10) * 10
    songs_by_decade = df['Decade'].value_counts().sort_index()
    if label== 'Birth_Year':
        title='Number of Songs per Artist Birth Year (Decades)'
    else:
        title='Number of Songs Released by Decade'
    plt.figure(figsize=(12, 6))
    ax = songs_by_decade.plot(kind='bar', color='coral')
    plt.title(title)
    plt.xlabel('Decade')
    plt.ylabel('Number of Songs')
    plt.xticks(rotation=45, ha='right')
    add_annotations(ax, songs_by_decade)
    plt.tight_layout()
    plt.show()

# Main function to plot all required graphs
def main():
    plot_songs_by_style(df)
    plot_songs_by_artist(df)
    plot_songs_by_decade(df,'Birth_Year')
    plot_songs_by_decade(df,'releaseYear')

if __name__ == "__main__":
    main()
