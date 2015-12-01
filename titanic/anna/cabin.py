# Для 200 пассажиров указаны номера кабин. Теоретически, запертые в нижних кабинах (G,F) 
# пассажиры третьего класса имели мало шансов на выживание.
# Аналитически, наибольший коэф. выживаемости у пасс. кабин средних рядов (см. 28-37)
# На Log.Reg результат 0.67. Неясно, как использовать результаты в общем анализе данных.
import pandas as pd

titanic = pd.read_csv("titanic.csv")
predictors = ["PassengerId","Survived","Pclass","Cabin"]
cabin=titanic.loc[:,predictors]
cabin=cabin.dropna(how='any')

def get_cabin2(name):
    cabin_search = name[0]
    if cabin_search:
        return cabin_search
    return ""    
    
n_cabin = cabin["Cabin"].apply(get_cabin2)
cabin["n_cabin"] = n_cabin
title_mapping = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8}
for k,v in title_mapping.items():
    n_cabin[n_cabin == k] = v
    cabin["n_cabin"] = n_cabin
tclass = cabin.groupby(["n_cabin", "Survived"]).size().unstack()
tclass[2]=tclass[1]/tclass[0]
print (tclass)

#Survived   0   1         2
#n_cabin                   
#1          8   7  0.875000
#2         12  35  2.916667
#3         24  35  1.458333
#4          8  25  3.125000
#5          8  24  3.000000
#6          5   8  1.600000
#7          2   2  1.000000
#8          1 NaN       NaN

from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
pred=["Pclass","n_cabin"]
X = cabin[pred] 
y = cabin["Survived"]
alg = LogisticRegression()
scores=cross_validation.cross_val_score(alg, X, y, scoring=None,
                                            cv=3, n_jobs=1, verbose=0, fit_params=None, pre_dispatch='2*n_jobs')
print('finally ',round(scores.mean(), 2))