# 0. Подготовка данных
import pandas as pd
from pandas import Series,DataFrame
import numpy as np
from sklearn import metrics
from sklearn import preprocessing
titanic = pd.read_csv("titanic1.csv")
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "FamilySize","Pclass_mean","Sex_mean","Embarked_mean"]
X = titanic[predictors]
#X = preprocessing.normalize(X)
X = preprocessing.scale(X)
y = titanic["Survived"]

def f_class(model, X,y):
    model.fit(X, y)
    print(model)
    expected = y
    predicted = model.predict(X)
##    print(metrics.classification_report(expected, predicted))
##    print(metrics.confusion_matrix(expected, predicted))
##    print(model.score(X, y))
##    print("-----------------------------------------")
    return round(model.score(X, y), 2)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

a1 = GaussianNB()#0.79
a2 = KNeighborsClassifier(n_neighbors=3) #0.84
a3 = SVC()#0.89
a4 = LogisticRegression()#0.80
a5 = DecisionTreeClassifier()#0.97
algorithms=[a1, a2, a3, a4, a5]
d=[]
for i in algorithms:
	scores=f_class(i, X, y)
	d.append(scores)
print(d)
# 0 without norm   [0.79, 0.84, 0.89, 0.80, 0.97]
# 0 with notm      [0.74, 0.86, 0.68, 0.69, 0.97] у LG и svm сильно хуже
# 0 with norm stand[0.74, 0.87, 0.82, 0.78, 0.97]
# 0 with stand     [0.79, 0.88, 0.83, 0.79, 0.97] KN+DesTree

# extended predictors
#  without norm    [0.77, 0.84, 0.88, 0.80, 0.97]
# 0 with notm      [0.76, 0.87, 0.68, 0.69, 0.97] у LG и svm сильно хуже
# 0 with norm stand[0.76, 0.87, 0.82, 0.79, 0.97]
# 0 with stand     [0.79, 0.88, 0.83, 0.79, 0.97] KN+DesTree
