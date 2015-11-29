# загрузить отсортированные входные данные и установить предикторы
# получить оценки по алгоритмам
import pandas as pd
from pandas import Series,DataFrame
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn import cross_validation
import matplotlib.pyplot as plt

titanic = pd.read_csv("titanic.csv")

X_train = titanic.drop("Survived",axis=1)
predictors = ["Pclass"]
X = titanic[predictors] 
y = titanic["Survived"]

def f_class(model, X,y):
    scores=cross_validation.cross_val_score(model, X, y, scoring=None,
                                            cv=3, n_jobs=1, verbose=0, fit_params=None, pre_dispatch='2*n_jobs')
    return scores.mean()

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

a1 = GaussianNB()
a2 = KNeighborsClassifier(n_neighbors=3) 
a3 = SVC()
a4 = LogisticRegression()
a5 = DecisionTreeClassifier()
a6 = GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3)
a7 = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=8, min_samples_leaf=4)
 
algorithms=[a1, a2, a3, a4, a5, a6, a7]
d=[]
for model in algorithms:
    scores=cross_validation.cross_val_score(model, X, y, scoring=None,
                                            cv=3, n_jobs=1, verbose=0, fit_params=None, pre_dispatch='2*n_jobs')
    scores.mean()
    d.append(scores.mean())
x=np.arange(len(d))
plt.bar(x, d, label='Died')
plt.xticks(np.array(x)+0.37, ['KNeighbors', 'Gauss', 'SVC', 'Logistic', 'Tree',
                             'Forest', 'Boosting'])
plt.ylabel("Probability")
plt.xlabel("")
plt.show()
print(d)
