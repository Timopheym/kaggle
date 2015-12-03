# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 12:17:03 2015

@author: timopheym
"""
import pandas as pd
from sklearn.svm import SVC
from features import train, test
from sklearn import metrics
from sklearn import preprocessing
from sklearn import cross_validation

predictors = ["Pclass", "Sex", "Fare", "Title",
              "NameLength"]
test = pd.read_csv("anna/men_errors.csv")

X = train[predictors] 
y = train["Survived"]


# X = preprocessing.normalize(X)
X = preprocessing.scale(X)

Z = test[predictors]
y_z = test["Survived"]

clf = SVC(kernel="poly", 
          gamma=3,
          cache_size=500,
          # class_weight='balanced',
          C = 0.8) #nozzy data
clf.fit(X, y) 

predicted = clf.predict(X)

# ===============SUBMIT================================
predicted_Z = clf.predict(Z)
predictions = predicted_Z.astype(int)
submission = pd.DataFrame({
   "PassengerId": test["PassengerId"],
   "Survived": predictions
})
submission.to_csv("kaggle_svc.csv", index=False)
#======================================================

print("X: ", metrics.classification_report(y, predicted))
print("X: ",metrics.confusion_matrix(y, predicted))
print("X: ",clf.score(X, y))
scores = cross_validation.cross_val_score(clf, X, y, scoring=None,
                                          cv=3, n_jobs=1, verbose=0, 
                                          fit_params=None, 
                                          pre_dispatch='2*n_jobs')
            
print("Z: ", metrics.classification_report(y_z, predicted_Z))
print("Z: ",metrics.confusion_matrix(y_z, predicted_Z))
print("Z: ",clf.score(Z, y_z))
scores = cross_validation.cross_val_score(clf, Z, y_z, scoring=None,
                                          cv=3, n_jobs=1, verbose=0, 
                                          fit_params=None, 
                                          pre_dispatch='2*n_jobs')
                    
print('Z: validated', round(scores.mean(), 2))
   