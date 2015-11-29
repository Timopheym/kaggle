print("we start!")
# 0. Подготовка данных
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import numpy as np

titanic = pd.read_csv("titanic.csv")
tclass = titanic.groupby(["Pclass", "Survived"]).size().unstack()
print (tclass)

red, blue = '#B2182B', '#2166AC'

plt.subplot(121)
plt.bar([0, 1, 2], tclass[0], color=red, label='Died')
plt.bar([0, 1, 2], tclass[1], bottom=tclass[0], color=blue, label='Survived')
plt.xticks([0.5, 1.5, 2.5], ['1st Class', '2nd Class', '3rd Class'], rotation='horizontal')
plt.ylabel("Number")
plt.xlabel("")
plt.legend(loc='upper left')

#normalize each row by transposing, normalizing each column, and un-transposing
tclass = (1. * tclass.T / tclass.T.sum()).T

plt.subplot(122)
plt.bar([0, 1, 2], tclass[0], color=red, label='Died')
plt.bar([0, 1, 2], tclass[1], bottom=tclass[0], color=blue, label='Survived')
plt.xticks([0.5, 1.5, 2.5], ['1st Class', '2nd Class', '3rd Class'], rotation='horizontal')
plt.ylabel("Fraction")
plt.xlabel("")

plt.show()
##
##titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
##titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
##titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]
##import matplotlib.pyplot as plt
##t=titanic["Age"]
##y = titanic["Survived"]
##plt.plot(t, y, '.',label='"FamilySize')
##plt.show()

##titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
##titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
##titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
##titanic["Embarked"] = titanic["Embarked"].fillna("S")
##titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
##titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
##titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2
##titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]
##titanic.loc[titanic["Pclass"] == 1, "Pclass"] = 1/3
##titanic.loc[titanic["Pclass"] == 2, "Pclass"] = 2/3
#titanic.loc[titanic["Pclass"] == 3, "Pclass"] = 1

##X_train = titanic.drop("Survived",axis=1)
##predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "FamilySize"]
##X = titanic[predictors] 
##y = titanic["Survived"]
print("success!")
