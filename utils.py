import random


def get_recommendation_ib():
    """
    Returns a list of recommendation item-based.
    """
    recos = [random.randrange(1, 5) for x in range(5)]
    return recos


def get_recommendation_ub():
    """
    Returns a list of recommendation user-based.
    """
    recos = [random.randrange(1, 5) for x in range(5)]
    return recos
