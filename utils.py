import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import SVD, KNNBasic
from surprise import accuracy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import string

import random

def text_cleaning(text):
    text = "".join([char for char in text if char not in string.punctuation])
    return text

def create_soup(x):
        return x['C_Type'] + " " + x['Veg_Non'] + " " + x['Describe']

def get_recommendation_ib(item):
    """
    Returns a list of recommendation item-based.
    """
    food_data = pd.read_csv('data/food.csv')
    ratings_data = pd.read_csv("data/ratings.csv")
    food_data['Describe'] = food_data['Describe'].apply(text_cleaning)

    # Merge data
    merged_data = pd.merge(ratings_data, food_data, on='Food_ID')

    # Define a Reader
    reader = Reader(rating_scale=(1, 5))

    # Create Surprise Dataset
    data = Dataset.load_from_df(merged_data[['User_ID', 'Food_ID', 'Rating']], reader)

    # Train-test split
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    # feature engineering
    food_data['soup'] = food_data.apply(create_soup, axis=1)

    # Create TF-IDF vectorizer for content-based filtering
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(food_data['soup'])

    # Compute cosine similarity
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Function to get recommendations based on content
    idx = food_data.index[food_data['Name'] == item.lower()].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    food_indices = [i[0] for i in sim_scores]
    return food_data['Name'].iloc[food_indices].values



def get_recommendation_ub(user_ID):
    """
    Returns a list of recommendation user-based.
    """
    food_data = pd.read_csv('data/food.csv')
    ratings_data = pd.read_csv('data/ratings.csv')  

    merged_data = pd.merge(ratings_data, food_data, on='Food_ID')

    # Define a Reader
    reader = Reader(rating_scale=(1, 10))

    # Create Surprise Dataset
    data = Dataset.load_from_df(merged_data[['User_ID', 'Food_ID', 'Rating']], reader)

    # Train-test split
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    sim_options = {
        'name': 'cosine',
        'user_based': True
    }

    # Build the collaborative filtering model
    model = KNNBasic(sim_options=sim_options, n_factors=20, n_epochs=5)
    model.fit(trainset)


    food_ids = merged_data['Food_ID'].unique()

    # Get the list of food IDs rated by the user
    food_ids_user_rated = merged_data[merged_data['User_ID'] == user_ID]['Food_ID'].values

    # Get the list of food IDs not rated by the user
    food_ids_user_not_rated = [food_id for food_id in food_ids if food_id not in food_ids_user_rated]

    # Create a list of tuples in the format (food_id, user_id, rating) for all food IDs not rated by the user
    food_ids_user_not_rated = [(food_id, user_ID, 0) for food_id in food_ids_user_not_rated]

    # Predict ratings for all food IDs not rated by the user
    predictions = model.test(food_ids_user_not_rated)

    # Get top 10 predictions
    recommendations = []
    for food_id, user_id, rating, _, _ in predictions:
        recommendations.append((food_id, rating))

    recommendations.sort(key=lambda x: x[1], reverse=True)
    recommendations = recommendations[:5]

    # Get food names
    food_names = []
    for recommendation in recommendations:
        food_names.append(food_data[food_data['Food_ID'] == recommendation[0]]['Name'].values[0])

    print("Here are the foods.")
    print(food_names)
    return food_names
    # return food_names