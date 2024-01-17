import streamlit as st
import pandas as pd
import random

from utils import get_recommendation_ib, get_recommendation_ub

# Loading the base data
ratings = pd.read_csv("data/ratings.csv").dropna()
unique_users = [f"Utilisateur n°{int(x)}" for x in ratings.User_ID.unique()]

food = pd.read_csv("data/food.csv").dropna()
unique_food = [x.capitalize() for x in food.Name.unique()]


st.write("## IDMC Recommender Systems Project")
st.write("Ce projet vise à démontrer l'implémentation d'algorithmes de recommandation sur des données de plats.")
st.markdown("*Julia Crapanzano, Maxime Haurel, Jules Margaritta, Marion Schmitt*")

# st.image('resources/food_img.jpg', caption='Credit: Unsplash @lvnatikk', width=300)

# Tabs
tab_user_based, tab_content_based = st.tabs(["User-based", "Content-based"])

with tab_user_based:
    st.header("User-based recommender system")
    # small text description
    st.write("lorem ipsum dolor sit amet.")
    # select -> choose the user from which we want the recommendations
    selected_user = st.selectbox(
        "Sélectionnez l'utilisateur pour lequel vous souhaitez obtenir des recommendations.",
        unique_users
    )

    # ! delete this
    # st.write('You selected:', selected_user)
    # display the recommendations
    for i, rec in enumerate(get_recommendation_ub(selected_user)):
        st.write(f"{i+1}. {rec.capitalize()}")


with tab_content_based:
    st.header("Content-based recommender system")
    # small text description
    st.write("Ce système de recommandation prend en compte les ingrédients de la nourriture, le type (Healthy, Indian, etc.) ainsi que le statut végan.")
    # select -> choose the user from which we want the recommendations
    selected_food = st.selectbox(
        "Sélectionnez le repas pour lequel vous souhaitez obtenir des recommendations.",
        unique_food
    )

    # ! delete this
    # st.write('You selected:', selected_food)
    # display the recommendations
    st.markdown("### Recommendations suggérées:")

    for i, rec in enumerate(get_recommendation_ib(selected_food)):
        st.write(f"{i+1}. {rec.capitalize()}")



