import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read CSV File
df = pd.read_csv("movie_dataset.csv", low_memory=False)


# Select Features
features = ['keywords', 'cast', 'genres', 'director']


# Create a Count Matrix from the Combined Features
for feature in features:
    df[feature] = df[feature].fillna('') #fill all NaN values with blank string

def combine_features(row):
    return row['keywords'] + " " + row['cast'] + " " + row['genres'] + " " + row['director']

df["combined_features"] = df.apply(combine_features, axis=1) #combine all the 4 selected features into a single string


# Compute the Cosine Similarity Based on the Count Matrix
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])

# Get the Movie Title from the User
cosine_sim = cosine_similarity(count_matrix)

# Helper Functions - Use [0] to ensure to return a primitive value, not a Series
def get_title_from_index(index):
    return df.iloc[index]["title"]

def get_index_from_title(title):
    return df[df.title == title].index[0]

# Recommendation Logic
movie_user_likes = "Avatar"

try:
    movie_index = get_index_from_title(movie_user_likes)
    similar_movies = list(enumerate(cosine_sim[movie_index]))

    # Sort by similarity score (the second element in the tuple)
    sorted_similar_movies = sorted(similar_movies, key=lambda x:x[1], reverse=True)

    print(f"Top 5 movies similar to '{movie_user_likes}':\n")

    # Start from index 1 to skip the movie itself (since similarity with itself will be 1.0)
    for i in range(1, 6):
        print(f"{i}. {get_title_from_index(sorted_similar_movies[i][0])}")

except IndexError:
    print(f"Error: '{movie_user_likes}' not found in the dataset.")
except Exception as e:
    print(f"An error occurred: {e}")