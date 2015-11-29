# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 18:04:36 2015

@author: timopheym
"""

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from features import train, test

predictors = ["Pclass", "Fare", "Title", 
              "FamilyId", "FamilySize", "Age",
              "Embarked"]
              
grad_boost = GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3)
rand_forest = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=8, min_samples_leaf=4)
  
algorithms=[grad_boost, rand_forest]
d=[] #??

X = train[predictors]  
y = train["Survived"]
diff = []
for model in algorithms:
        model = model.fit(X,y) 
        p = pd.DataFrame(model.predict_proba(X)) # ndarray
#        foo = map(lambda xs: xs[0]+xs[1], p)
        z = model.predict(X)       # ndarray      
        #guesses = X.copy()
        guesses = X.copy()
        guesses["prob"]=p[1]
        guesses["predicted_class"]=z
        guesses["right_predicted"]= pd.DataFrame(list(map(lambda x: not x, np.array(y) - np.array(z))))
        wrong = guesses[guesses["right_predicted"] == False]
        diff.append(wrong)        
        

    
