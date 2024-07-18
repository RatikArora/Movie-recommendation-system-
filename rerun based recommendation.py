import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
import joblib

# Load data
data = pd.read_csv('ratings.csv')
print(data.head())

# Create user-item matrix
user_item_matrix = data.pivot(index='userId', columns='movieId', values='rating').fillna(0)
print(user_item_matrix)

# Load the movies data
movies = pd.read_csv('movies.csv')
print(movies.head())

# Check if model file exists to avoid retraining
import os

if os.path.exists('svd_model.pkl'):
    # Load the SVD model
    svd = joblib.load('svd_model.pkl')
    latent_matrix = svd.transform(user_item_matrix)
    latent_matrix_transpose = svd.components_
else:
    # Train the SVD model
    svd = TruncatedSVD(n_components=50)
    latent_matrix = svd.fit_transform(user_item_matrix)
    latent_matrix_transpose = svd.components_

    # Save the SVD model
    joblib.dump(svd, 'svd_model.pkl')

# Reconstruct the matrix
reconstructed_matrix = np.dot(latent_matrix, latent_matrix_transpose)

# Function to get top N recommendations
def get_top_n_recommendations(user_id, reconstructed_matrix, user_item_matrix, movies, n=10):
    user_ratings = reconstructed_matrix[user_id - 1]
    original_ratings = user_item_matrix.loc[user_id]
    already_rated = original_ratings[original_ratings > 0].index
    recommended_items = [(i, user_ratings[i]) for i in range(len(user_ratings)) if i not in already_rated]
    recommended_items.sort(key=lambda x: x[1], reverse=True)
    top_recommendations = recommended_items[:n]
    
    # Get movie names and genres
    top_recommendations_with_details = [
        (
            movies.loc[movies['movieId'] == item_id, 'movieName'].values[0], 
            movies.loc[movies['movieId'] == item_id, 'genre'].values[0],
            rating
        )
        for item_id, rating in top_recommendations
    ]
    
    return top_recommendations_with_details

# Function to get user's favorite genre
def get_user_favorite_genre(user_id, user_item_matrix, movies):
    user_ratings = user_item_matrix.loc[user_id]
    watched_movies = user_ratings[user_ratings > 0].index
    watched_movies_with_ratings = user_ratings[user_ratings > 0]
    
    # Merge watched movies with their genres
    watched_movies_genres = movies[movies['movieId'].isin(watched_movies)][['movieId', 'genre']]
    watched_movies_genres = watched_movies_genres.set_index('movieId')
    watched_movies_genres = watched_movies_genres.loc[watched_movies_with_ratings.index]
    watched_movies_genres['rating'] = watched_movies_with_ratings.values
    
    # Explode genres and calculate average rating for each genre
    watched_movies_genres['genre'] = watched_movies_genres['genre'].str.split('|')
    exploded_genres = watched_movies_genres.explode('genre')
    genre_ratings = exploded_genres.groupby('genre')['rating'].mean().sort_values(ascending=False)
    
    return genre_ratings.index[0]  # Return the favorite genre

# Get top 10 recommendations for a user
user_id = int(input("Enter the user Id : "))
top_recommendations = get_top_n_recommendations(user_id, reconstructed_matrix, user_item_matrix, movies, n=10)

# Print the recommendations
print(f'Top 10 recommendations for User {user_id}:')
for movie_name, genre, rating in top_recommendations:
    print(f'Movie: {movie_name}, Genre: {genre}, Estimated Rating: {rating}')

# Get user's favorite genre
favorite_genre = get_user_favorite_genre(user_id, user_item_matrix, movies)
print(f"User {user_id}'s favorite genre: {favorite_genre}")
