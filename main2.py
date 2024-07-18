import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

# Load data
data = pd.read_csv('ratings.csv')
print(data.head())

# Create user-item matrix
user_item_matrix = data.pivot(index='userId', columns='movieId', values='rating').fillna(0)
print(user_item_matrix)

# Load the movies data
movies = pd.read_csv('movies.csv')
print(movies.head())

# Check if similarity matrix file exists to avoid recalculating
if os.path.exists('item_similarity_matrix.pkl'):
    # Load the item similarity matrix
    item_similarity_matrix = joblib.load('item_similarity_matrix.pkl')
else:
    # Calculate item-item similarity matrix using cosine similarity
    item_similarity_matrix = cosine_similarity(user_item_matrix.T)
    
    # Save the item similarity matrix
    joblib.dump(item_similarity_matrix, 'item_similarity_matrix.pkl')

# Convert item similarity matrix to a DataFrame for easier manipulation
item_similarity_df = pd.DataFrame(item_similarity_matrix, index=user_item_matrix.columns, columns=user_item_matrix.columns)

# Function to get top N recommendations using item-based collaborative filtering
def get_top_n_recommendations(user_id, user_item_matrix, item_similarity_df, movies, n=10):
    user_ratings = user_item_matrix.loc[user_id]
    already_rated = user_ratings[user_ratings > 0].index
    
    # Calculate predicted ratings for all items
    predicted_ratings = user_item_matrix.dot(item_similarity_df) / np.array([np.abs(item_similarity_df).sum(axis=1)])
    
    # Filter out items already rated by the user
    recommended_items = [(i, predicted_ratings.loc[user_id, i]) for i in predicted_ratings.columns if i not in already_rated]
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
        if not movies.loc[movies['movieId'] == item_id].empty
    ]
    
    return top_recommendations_with_details

# Precompute and store recommendations for all users
user_ids = user_item_matrix.index.tolist()
recommendations = {}
for user_id in user_ids:
    recommendations[user_id] = get_top_n_recommendations(user_id, user_item_matrix, item_similarity_df, movies, n=10)

# Save recommendations to disk
joblib.dump(recommendations, 'recommendations.pkl')

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
    
    return genre_ratings.index[0]  

# Load precomputed recommendations
recommendations = joblib.load('recommendations.pkl')

# Get recommendations and favorite genre for a specific user
user_id = int(input("Enter the user Id : "))
top_recommendations = recommendations[user_id]

# Print the recommendations
print(f'Top 10 recommendations for User {user_id}:')
for movie_name, genre, rating in top_recommendations:
    print(f'Movie: {movie_name}, Genre: {genre}, Estimated Rating: {rating}')

# Get user's favorite genre
favorite_genre = get_user_favorite_genre(user_id, user_item_matrix, movies)
print(f"User {user_id}'s favorite genre: {favorite_genre}")
