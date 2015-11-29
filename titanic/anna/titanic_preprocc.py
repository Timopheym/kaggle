print("we start!")
# 0. Подготовка данных
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import numpy as np

titanic = pd.read_csv("titanic.csv")

titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
titanic["Embarked"] = titanic["Embarked"].fillna("S")
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2
titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]

titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))
import re
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""
titles = titanic["Name"].apply(get_title)
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
for k,v in title_mapping.items():
    titles[titles == k] = v
#print(pd.value_counts(titles))
titanic["Title"] = titles

# Feature_selection
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title"]
selector = SelectKBest(f_classif, k=5)
selector.fit(titanic[predictors], titanic["Survived"])
# Get the raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)
import matplotlib.pyplot as plt
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.show()
predictors = ["Pclass", "Sex", "Fare", "Title"]


##titanic0=titanic
##
##titanic["Pclass_mean"]=titanic["Pclass"]
##titanic.loc[titanic["Pclass_mean"] == 1, "Pclass_mean"] = 0.62
##titanic.loc[titanic["Pclass_mean"] == 2, "Pclass_mean"] = 0.42
##titanic.loc[titanic["Pclass_mean"] == 3, "Pclass_mean"] = 0.25
##titanic["Sex_mean"]=titanic["Sex"]
##titanic.loc[titanic["Sex_mean"] == 0, "Sex_mean"] = 0.2
##titanic.loc[titanic["Sex_mean"] == 1, "Sex_mean"] = 0.8
##titanic["Embarked_mean"] = titanic["Embarked"]
##titanic.loc[titanic["Embarked_mean"] == 0, "Embarked_mean"] = 0.35
##titanic.loc[titanic["Embarked_mean"] == 1, "Embarked_mean"] = 0.4
##titanic.loc[titanic["Embarked_mean"] == 2, "Embarked_mean"] = 0.55
##
##titanic1=titanic
##
##titanic0.to_csv("titanic0.csv")
##titanic1.to_csv("titanic1.csv")

print("success!")
