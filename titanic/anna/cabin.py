import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import numpy as np

titanic = pd.read_csv("titanic.csv")
cabin=titanic.loc[:,["PassengerId","Survived","Pclass","Cabin"]]
cabin=cabin.dropna(how='any')
print(cabin.head())

import re
def get_cabin(name):
    cabin_search = re.search('[A-G]', name)
    if cabin_search:
        print(cabin_search)
        return cabin_search
    return ""
n_cabin = cabin["Cabin"].apply(get_cabin)
cabin["n_cabin"] = n_cabin
print(cabin.head())
title_mapping = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
for k,v in title_mapping.items():
    n_cabin[n_cabin == k] = v
cabin["n_cabin"] = n_cabin
print(cabin.head())
