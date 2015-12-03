# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import svm
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn import metrics
from features import train

# Feature_selection
from sklearn.feature_selection import SelectKBest, f_classif
predictors = ["Pclass", "Sex", "Age", 
              "SibSp", "Parch", "Fare", "NameLength",
              "Embarked", "FamilySize", "Title", "FamilyId"]
X = train[predictors] 
y = train["Survived"]
selector = SelectKBest(f_classif, k=5)
selector.fit(X, y)
scores = -np.log10(selector.pvalues_)
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.show()
#predictors = ["Pclass", "Sex", "Fare", "Title",
#              "NameLength"]

#predictors = ["Pclass", "Fare", "Title", 
#              "FamilyId", "FamilySize", "Age",
#              "NameLength", "Embarked"]

#X = train[predictors] 
#
#normalized_X = preprocessing.normalize(X)
#standardized_X = preprocessing.scale(X)

#a1 = SVC()
#model = svm.SVC(kernel='poly',gamma=3)
#a3 = svm.SVC(kernel='linear',gamma=3)
#a4 = svm.SVC(kernel='rbf',gamma=3)
#a5 = svm.SVC(gamma=3) 
#a6 = svm.SVC(gamma=4)
#a7 = svm.SVC(gamma=5)
 
#algorithms=[a2]#, a2]#, a3, a4, a5, a6, a7]
#d=[]
#model=a2
#model.fit(X, y)
#print(model)
#expected = y
#predicted = model.predict(X)
#print(metrics.classification_report(expected, predicted))
#print(metrics.confusion_matrix(expected, predicted))
#print(model.score(X, y))

#for model in algorithms:
#    scores=cross_validation.cross_val_score(model, X, y, scoring=None,
#                                            cv=3, n_jobs=1, verbose=0, fit_params=None, pre_dispatch='2*n_jobs')
#                    
#    d.append(round(scores.mean(), 2))
#    
#x=np.arange(len(d))
#algo_names = [algorithms]
#plt.bar(x, d, label='Died')
#plt.xticks(np.array(x)+0.37, algo_names)
#plt.ylabel("Probability")
#plt.xlabel("")
#plt.show()
#
#print(d)
#print("Winner is: ", algo_names[d.index(max(d))], max(d))