import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
    df = pd.read_csv('data/food.csv')
    df['Describe'] = df['Describe'].apply(text_cleaning)
    
    df['soup'] = df.apply(create_soup, axis=1)
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(df['soup'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    df = df.reset_index()
    indices = pd.Series(df.index, index=df['Name'])


    idx = indices[item.lower()]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:6]

    food_indices = [i[0] for i in sim_scores]
    return df['Name'].iloc[food_indices].values
    # return df['Name'].iloc[food_indices]



def get_recommendation_ub():
    """
    Returns a list of recommendation user-based.
    """
    recos = [random.randrange(1, 5) for x in range(5)]
    print(recos)
    return recos
