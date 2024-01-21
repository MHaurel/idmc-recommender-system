import pandas as pd
from surprise import Dataset, Reader, dump
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import string
import os

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
    user_id = int(user_ID.split("°")[-1])

    food_data = pd.read_csv('data/food.csv')
    ratings_data = pd.read_csv("data/ratings.csv")
    food_data['Describe'] = food_data['Describe'].apply(text_cleaning)
    merged_data = pd.merge(ratings_data, food_data, on='Food_ID')

    _, algo = dump.load(os.path.join('models', 'ub_model.dump'))

    def get_user_recommendations(user_id, algo, n=5):
        predictions = []
        for item in merged_data['Food_ID'].unique():
            predictions.append(algo.predict(user_id, item))
        
        best_items = sorted([(prediction.iid, prediction.est) for prediction in predictions], key=lambda x: x[1])[::-1][:n]
        best_items_idx = [x for x, y in best_items]
        rec_items = food_data.loc[best_items_idx]
        return rec_items['Name']

    recommendations = get_user_recommendations(user_id, algo, n=5).values

    return recommendations