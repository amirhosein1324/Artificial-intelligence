import numpy as np
import pandas as pd


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df_ratings=pd.read_csv("/content/ratings.csv" )
df_movies_metadata=pd.read_csv("/content/movies_metadata.csv")

df_movies_metadata.head()

df_ratings.head()

movie_user_ratings = df_ratings.groupby('movieId').agg({'userId': list, 'rating': list})

movie_user_ratings['selection_count'] = movie_user_ratings['userId'].apply(len)

movie_user_ratings = movie_user_ratings.sort_values(by='selection_count', ascending=False)


movie_user_ratings = movie_user_ratings.reset_index()


top_1000_movies = movie_user_ratings.head(1000)

top_1000_movies['user_rating'] = top_1000_movies.apply(lambda row: list(zip(row['userId'], row['rating'])), axis=1)

print(top_1000_movies[['movieId', 'user_rating']])

dataframe = pd.DataFrame(data)
dataframe

report_df = df_ratings.pivot_table(index='userId', columns='movieId', values='rating' , fill_value = 0)
print(report_df)



