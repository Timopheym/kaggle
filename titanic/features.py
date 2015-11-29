# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 16:09:36 2015

@author: timopheym
"""

import pandas as pd
import numpy as np
import re
#from sklearn.feature_selection import SelectKBest, f_classif
 
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
#AGE and SEX

std_age = train["Age"].std()
mean_age = train["Age"].mean()
train["Age"][np.isnan(train["Age"])] = np.random.randint(
                                                     mean_age - std_age,
                                                     mean_age + std_age, 
                                                     size = train["Age"]
                                                             .isnull().sum())
train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1

#Get children 
train.loc[train["Age"] < 16, "Sex"] = 2

train["Embarked"] = train["Embarked"].fillna("S")
train.loc[train["Embarked"] == "S", "Embarked"] = 0
train.loc[train["Embarked"] == "C", "Embarked"] = 1
train.loc[train["Embarked"] == "Q", "Embarked"] = 2

train["FamilySize"] = train["SibSp"] + train["Parch"]

train["NameLength"] = train["Name"].apply(lambda x: len(x))

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""
titles = train["Name"].apply(get_title)
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
for k,v in title_mapping.items():
    titles[titles == k] = v
#print(pd.value_counts(titles))
train["Title"] = titles

import operator
family_id_mapping = {}
def get_family_id(row):
    last_name = row["Name"].split(",")[0]
    family_id = "{0}{1}".format(last_name, row["FamilySize"])
    if family_id not in family_id_mapping:
        if len(family_id_mapping) == 0:
            current_id = 1
        else:
            # Get the maximum id from the mapping and add one to it if we don't have an id
            current_id = (max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1)
        family_id_mapping[family_id] = current_id
    return family_id_mapping[family_id]
family_ids = train.apply(get_family_id, axis=1)
family_ids[train["FamilySize"] < 3] = -1
#print(pd.value_counts(family_ids))
train["FamilyId"] = family_ids

# Test data

test["Age"]=test["Age"].fillna(train["Age"].median())

test.loc[test["Sex"]=="male", "Sex"]=0
test.loc[test["Sex"]=="female", "Sex"]=1

test["Embarked"]=test["Embarked"].fillna("S")
test.loc[test["Embarked"]=="S", "Embarked"]=0
test.loc[test["Embarked"]=="C", "Embarked"]=1
test.loc[test["Embarked"]=="Q", "Embarked"]=2

test["Fare"]=test["Fare"].fillna(test["Fare"].median())

titles = test["Name"].apply(get_title)
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2, "Dona": 10}
for k,v in title_mapping.items():
    titles[titles == k] = v
test["Title"] = titles
#print(pandas.value_counts(test["Title"]))
test["FamilySize"] = test["SibSp"] + test["Parch"]
#print(family_id_mapping)
family_ids = test.apply(get_family_id, axis=1)
family_ids[test["FamilySize"] < 3] = -1
test["FamilyId"] = family_ids
test["NameLength"]=test["Name"].apply(lambda x: len(x))

