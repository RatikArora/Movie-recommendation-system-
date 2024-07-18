Movie Recommendation System

This repository implements a movie recommendation system using Python and the MovieLens 1M dataset.

Features:

Generates personalized movie recommendations based on user ratings and collaborative filtering using Truncated SVD.
Recommends movies the user hasn't rated yet.
Provides the user's estimated rating for each recommended movie.
Identifies the user's favorite genre based on their watched movies and ratings.
Requirements:

Python 3.x
pandas
numpy
scikit-learn
joblib
Installation:

Clone this repository: git clone https://github.com/your-username/movie-recommendation-system.git
Create a virtual environment (recommended): python -m venv venv
Activate the virtual environment: source venv/bin/activate (Linux/macOS) or venv\Scripts\activate.bat (Windows)
Install dependencies: pip install -r requirements.txt
Usage:

Download the MovieLens 1M dataset from https://grouplens.org/datasets/movielens/1m/
Extract the downloaded archive and place the ratings.csv and movies.csv files in the same directory as this code.
Run the script: python movie_recommendation_system.py
Enter the User ID when prompted. The script will display the top 10 recommendations and the user's favorite genre.
Model Persistence:

The system saves the trained Truncated SVD model as svd_model.pkl to avoid retraining on every run.
This improves performance and efficiency.
Code Structure:

movie_recommendation_system.py: The main script containing data loading, model training/loading, recommendation generation, and user favorite genre identification functions.
Further Enhancements (Optional):

Implement a user interface (e.g., web application) for a more interactive experience.
Explore different recommendation algorithms like content-based filtering or hybrid approaches.
Incorporate movie descriptions, ratings from other users, or additional user information for more refined recommendations.
Contributing:

We welcome contributions to improve this project. Please feel free to fork the repository, make changes, and create pull requests.

License:

This project is licensed under the MIT License (see LICENSE file).

Dataset:

This project uses the MovieLens 1M dataset, available at https://grouplens.org/datasets/movielens/1m/.
