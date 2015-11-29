# 0. Подготовка данных
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

titanic = pd.read_csv("titanic.csv")
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
titanic["Embarked"] = titanic["Embarked"].fillna("S")
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2
titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]
print(titanic["Age"]<10)
# Подготовка
titanic_test = pd.read_csv("titanic_test.csv")
titanic_test["Age"]=titanic_test["Age"].fillna(titanic["Age"].median())
titanic_test.loc[titanic_test["Sex"]=="male", "Sex"]=0
titanic_test.loc[titanic_test["Sex"]=="female", "Sex"]=1
titanic_test["Embarked"]=titanic_test["Embarked"].fillna("S")
titanic_test.loc[titanic_test["Embarked"]=="S", "Embarked"]=0
titanic_test.loc[titanic_test["Embarked"]=="C", "Embarked"]=1
titanic_test.loc[titanic_test["Embarked"]=="Q", "Embarked"]=2
titanic_test["Fare"]=titanic_test["Fare"].fillna(titanic_test["Fare"].median())
titanic_test["FamilySize"] = titanic_test["SibSp"] + titanic_test["Parch"]


X_train = titanic.drop("Survived",axis=1)
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "FamilySize"]
predictors = ["Age"]
X = titanic[predictors] 
y = titanic["Survived"]
print()

from sklearn import metrics

def f_class(model, X,y):
    model.fit(X, y)
    print(model)
    # make predictions
    expected = y
    predicted = model.predict(X)
    # summarize the fit of the model
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))
    print(model.score(X, y))
    print("-----------------------------------------")
    return model.score(X, y)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

a1 = GaussianNB()#0.79
a2 = KNeighborsClassifier(n_neighbors=3) #0.84
a3 = SVC()#0.89
a4 = LogisticRegression()#0.80
a5 = DecisionTreeClassifier()#0.97
a6 = GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3)
a7 = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=8, min_samples_leaf=4)
 
algorithms=[a1, a2, a3, a4, a5, a6, a7]
d=[]
for i in algorithms:
	scores=f_class(i, X, y)
	d.append(scores)
print(d)

##from sklearn.linear_model import Ridge
##from sklearn.grid_search import GridSearchCV
### prepare a range of alpha values to test
##alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
### create and fit a ridge regression model, testing each alpha
##model = Ridge()
##grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))
##grid.fit(X, y)
##print(grid)
### summarize the results of the grid search
##print(grid.best_score_)
##print(grid.best_estimator_.alpha)
###0.37
##
##from scipy.stats import uniform as sp_rand
##from sklearn.linear_model import Ridge
##from sklearn.grid_search import RandomizedSearchCV
### prepare a uniform distribution to sample for the alpha parameter
##param_grid = {'alpha': sp_rand()}
### create and fit a ridge regression model, testing random alpha values
##model = Ridge()
##rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100)
##rsearch.fit(X, y)
##print(rsearch)
### summarize the results of the random parameter search
##print(rsearch.best_score_)
##print(rsearch.best_estimator_.alpha)
##print(rsearch(type))
