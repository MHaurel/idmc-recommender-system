import streamlit as st
import pandas as pd

from utils import get_recommendation_ib, get_recommendation_ub


@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df


df = load_data("data/ratings.csv")
df_matrix = load_data("data/ratings_matrix.csv")
st.dataframe(df_matrix)

max_user = int(df.User_ID.max())


st.title("Recommender systems demonstration")
st.markdown("> A use case on the Food Recommendation dataset")
st.markdown("dataset link")

user_number = st.slider(
    'Which user would you like to test the algorithm on ?',
    1, max_user
)

option = st.selectbox(
    'Which algorithm would you use ?',
    ('Item-based', 'User-based'))

st.write('You selected:', option)

if option == "Item-based":
    recos = get_recommendation_ib()
elif option == "User-based":
    recos = get_recommendation_ub()

for reco in recos:
    st.write(str(reco))
