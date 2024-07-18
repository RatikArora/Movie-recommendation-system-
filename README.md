# Movie Recommendation System

This project implements a collaborative filtering-based movie recommendation system using Singular Value Decomposition (SVD). The system recommends movies to users based on their past ratings and identifies users' favorite genres. The recommendation process is optimized to avoid retraining the model every time, ensuring faster performance.

## Table of Contents

1. [Introduction](#introduction)
2. [Datasets](#datasets)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Code Explanation](#code-explanation)
6. [Results](#results)
7. [License](#license)

## Introduction

This project uses collaborative filtering to recommend movies to users. Collaborative filtering leverages the idea that similar users will have similar preferences. We use the SVD algorithm to reduce the dimensionality of the user-item matrix and make predictions for items that a user has not rated yet. Additionally, the system can identify a user's favorite genre based on their past ratings.

## Datasets

We use two datasets for this project:
1. **Ratings Dataset**: Contains user ratings for different movies.
2. **Movies Dataset**: Contains movie metadata, including movie names and genres.

You can download the datasets from the [MovieLens website](https://grouplens.org/datasets/movielens/1m/).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/RatikArora/movie-recommendation-system-.git
   cd movie-recommendation-system-
   ```

2. Install the required packages:
   ```bash
   pip install pandas numpy scikit-learn joblib
   ```

3. Download and place the datasets (`ratings.csv` and `movies.csv`) in the project directory.

## Usage

1. Run the main script:
   ```bash
   python main.py
   ```

2. Enter the user ID when prompted to get movie recommendations and the user's favorite genre.

## Code Explanation

### Importing Libraries and Loading Data

```python
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
import joblib
import os

# Load data
data = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')
```

### Creating the User-Item Matrix

```python
user_item_matrix = data.pivot(index='userId', columns='movieId', values='rating').fillna(0)
```

### Checking for Existing Model

```python
if os.path.exists('svd_model.pkl'):
    svd = joblib.load('svd_model.pkl')
    latent_matrix = svd.transform(user_item_matrix)
    latent_matrix_transpose = svd.components_
else:
    svd = TruncatedSVD(n_components=50)
    latent_matrix = svd.fit_transform(user_item_matrix)
    latent_matrix_transpose = svd.components_
    joblib.dump(svd, 'svd_model.pkl')
```

### Reconstructing the Matrix

```python
reconstructed_matrix = np.dot(latent_matrix, latent_matrix_transpose)
```

### Getting Top N Recommendations

```python
def get_top_n_recommendations(user_id, reconstructed_matrix, user_item_matrix, movies, n=10):
    user_ratings = reconstructed_matrix[user_id - 1]
    original_ratings = user_item_matrix.loc[user_id]
    already_rated = original_ratings[original_ratings > 0].index
    recommended_items = [(i, user_ratings[i]) for i in range(len(user_ratings)) if i not in already_rated]
    recommended_items.sort(key=lambda x: x[1], reverse=True)
    top_recommendations = recommended_items[:n]

    top_recommendations_with_details = [
        (
            movies.loc[movies['movieId'] == item_id, 'movieName'].values[0], 
            movies.loc[movies['movieId'] == item_id, 'genre'].values[0],
            rating
        )
        for item_id, rating in top_recommendations
    ]
    
    return top_recommendations_with_details
```

### Getting User's Favorite Genre

```python
def get_user_favorite_genre(user_id, user_item_matrix, movies):
    user_ratings = user_item_matrix.loc[user_id]
    watched_movies = user_ratings[user_ratings > 0].index
    watched_movies_with_ratings = user_ratings[user_ratings > 0]

    watched_movies_genres = movies[movies['movieId'].isin(watched_movies)][['movieId', 'genre']]
    watched_movies_genres = watched_movies_genres.set_index('movieId')
    watched_movies_genres = watched_movies_genres.loc[watched_movies_with_ratings.index]
    watched_movies_genres['rating'] = watched_movies_with_ratings.values

    watched_movies_genres['genre'] = watched_movies_genres['genre'].str.split('|')
    exploded_genres = watched_movies_genres.explode('genre')
    genre_ratings = exploded_genres.groupby('genre')['rating'].mean().sort_values(ascending=False)
    
    return genre_ratings.index[0]
```

### Running the Script

```python
user_id = int(input("Enter the user Id : "))
top_recommendations = get_top_n_recommendations(user_id, reconstructed_matrix, user_item_matrix, movies, n=10)

print(f'Top 10 recommendations for User {user_id}:')
for movie_name, genre, rating in top_recommendations:
    print(f'Movie: {movie_name}, Genre: {genre}, Estimated Rating: {rating}')

favorite_genre = get_user_favorite_genre(user_id, user_item_matrix, movies)
print(f"User {user_id}'s favorite genre: {favorite_genre}")
```

## Results

Upon running the script, you will be prompted to enter a user ID. The script will then output the top 10 movie recommendations for that user along with their favorite genre.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
