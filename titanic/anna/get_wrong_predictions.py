# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 18:04:36 2015

@author: timopheym
"""

#

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt

from features import train

predictors = ["Pclass", "Sex", "Fare", "Title", 
              "FamilyId", "FamilySize", "Age",
              "NameLength", "Embarked"]
              
grad_boost = GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3)
rand_forest = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=8, min_samples_leaf=4)
  
algorithms=[grad_boost, rand_forest]

X = train[predictors]  
y = train["Survived"]
diff = []
#for model in algorithms:
#        model = model.fit(X,y) 
#        p = pd.DataFrame(model.predict_proba(X)) # ndarray
##        foo = map(lambda xs: xs[0]+xs[1], p)
#        z = model.predict(X)       # ndarray      
#        guesses = X.copy()
#        guesses["prob"]=p[1]
#        guesses["predicted_class"]=z
#        guesses["right_predicted"]= pd.DataFrame(list(map(lambda x: not x, np.array(y) - np.array(z))))
#        wrong = guesses[guesses["right_predicted"] == False]
#        diff.append(wrong)        
        
for model in algorithms:
        model = model.fit(X,y) 
        p = pd.DataFrame(model.predict_proba(X)) # ndarray
#        foo = map(lambda xs: xs[0]+xs[1], p)
        z = model.predict(X)       # ndarray      
        guesses = X.copy()
        guesses["prob"]=p[1]
        guesses["predicted_class"]=z
        diff.append(guesses)   

results=diff[0]
results["prob2","predicted_class2"]=diff[1].loc[:,["prob","predicted_class"]]

#guesses["right_predicted"]= pd.DataFrame(list(map(lambda x: not x, np.array(y) - np.array(z))))
#wrong = guesses[guesses["right_predicted"] == False]
#
#tclass = c.groupby(["Pclass", "predicted_class"]).size().unstack()
#
#red, blue = '#B2182B', '#2166AC'
#
#plt.subplot(121)
#plt.bar([0, 1, 2], tclass[0], color=red, label='Died')
#plt.bar([0, 1, 2], tclass[1], bottom=tclass[0], color=blue, label='Survived')
#plt.xticks([0.5, 1.5, 2.5], ['1st Class', '2nd Class', '3rd Class'], rotation='horizontal')
#plt.ylabel("Number")
#plt.xlabel("")
#plt.legend(loc='upper left')
#
##normalize each row by transposing, normalizing each column, and un-transposing
#tclass = (1. * tclass.T / tclass.T.sum()).T
#
#plt.subplot(122)
#plt.bar([0, 1, 2], tclass[0], color=red, label='Died')
#plt.bar([0, 1, 2], tclass[1], bottom=tclass[0], color=blue, label='Survived')
#plt.xticks([0.5, 1.5, 2.5], ['1st Class', '2nd Class', '3rd Class'], rotation='horizontal')
#plt.ylabel("Fraction")
#plt.xlabel("")
#
#plt.show()
#
#tclass = c.groupby(["Sex", "predicted_class"]).size().unstack()
#
#red, blue = '#B2182B', '#2166AC'
#
#plt.subplot(121)
#plt.bar([0, 1, 2], tclass[0], color=red, label='Died')
#plt.bar([0, 1, 2], tclass[1], bottom=tclass[0], color=blue, label='Survived')
#plt.xticks([0.5, 1.5, 2.5], ['male', 'female', 'children'], rotation='horizontal')
#plt.ylabel("Number")
#plt.xlabel("")
#plt.legend(loc='upper left')
#
##normalize each row by transposing, normalizing each column, and un-transposing
#tclass = (1. * tclass.T / tclass.T.sum()).T
#
#plt.subplot(122)
#plt.bar([0, 1, 2], tclass[0], color=red, label='Died')
#plt.bar([0, 1, 2], tclass[1], bottom=tclass[0], color=blue, label='Survived')
#plt.xticks([0.5, 1.5, 2.5], ['male', 'female', 'children'], rotation='horizontal')
#plt.ylabel("Fraction")
#plt.xlabel("")
