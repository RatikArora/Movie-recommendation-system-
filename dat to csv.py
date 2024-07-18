import pandas as pd

# Replace 'ratings.dat' with your actual .dat file path
dat_file = 'ml-10m/ml-10M100K/movies.dat'

# Define the columns based on the structure of your .dat file
columns = ['movieId', 'movieName', 'genre']

# Read .dat file into a DataFrame
with open(dat_file, 'r' ,encoding="utf-8") as f:
    data = f.readlines()

data = [line.strip().split('::') for line in data]

# Convert data into a DataFrame
df = pd.DataFrame(data, columns=columns)

# Convert types if necessary (e.g., userId, movieId to integers)
df['movieId'] = df['movieId'].astype(int)
df['movieName'] = df['movieName'].astype(str)
df['genre'] = df['genre'].astype(str)

# Save as .csv file
csv_file = 'movies.csv'
df.to_csv(csv_file, index=False)

print(f'Converted {dat_file} to {csv_file}')
