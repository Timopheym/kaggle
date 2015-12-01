# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 18:04:36 2015

@author: timopheym
"""
# Смотрим ошибки на алгоритах RandomForest и GradientBoosting
# Оба излишне убивают мужиков(до 100%) и 1 класс (до 90%)
# Оба излишне позволяют выжить ж.(до 90%) и детям (до 75%)
# Общая ошибка 88 шт. (из 90 для RF и 130 для GB)

# Прога рисует графики результатов работы алгоритмы для ошибок одного из алг. 
# и пересечения ошибок обоих алгоритмов

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt

from features import train

import matplotlib.pyplot as plt
def picturing(h):
    a=h["prob_GB"]
    plt.plot(a,'.', label='errors')
    plt.legend()
    plt.show()
    
    tclass = h.groupby(["Pclass", "suv_GB"]).size().unstack()
    red, blue = '#B2182B', '#2166AC'
    tclass = (1. * tclass.T / tclass.T.sum()).T
    
    plt.bar([0, 1, 2], tclass[0], color=red, label='Died')
    plt.bar([0, 1, 2], tclass[1], bottom=tclass[0], color=blue, label='Survived')
    plt.xticks([0.5, 1.5, 2.5], ['1st Class', '2nd Class', '3rd Class'], rotation='horizontal')
    plt.show()
    
    tclass = h.groupby(["Sex", "suv_GB"]).size().unstack()
    tclass = (1. * tclass.T / tclass.T.sum()).T
    plt.bar([0, 1, 2], tclass[0], color=red, label='Died')
    plt.bar([0, 1, 2], tclass[1], bottom=tclass[0], color=blue, label='Survived')
    plt.xticks([0.5, 1.5, 2.5], ['male', 'female', 'children'], rotation='horizontal')
    plt.show()


predictors = ["Pclass", "Sex", "Fare", "Title", 
              "FamilyId", "FamilySize", "Age",
              "NameLength", "Embarked"]
              
X = train[predictors]  
y = train["Survived"]
Z = train.copy()   

grad_boost = GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3)
rand_forest = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=8, min_samples_leaf=4)
       
model = grad_boost.fit(X,y) 
p = pd.DataFrame(model.predict_proba(X)) # ndarray
z = model.predict(X)       # ndarray      
Z["prob_GB"]=p[1]
Z["suv_GB"]=z
Z["fault_GB"]=z==y # если ошибка, то знач. False
S=Z[Z["fault_GB"]==False]
picturing(S)

model = rand_forest.fit(X,y) 
p = pd.DataFrame(model.predict_proba(X)) # ndarray
z = model.predict(X)       # ndarray      
Z["prob_RF"]=p[1]
Z["suv_RF"]=z
Z["fault_RF"]=z==y # если ошибка, то знач. False
Z["un"]=Z["fault_RF"]==Z["fault_GB"]  # класс по обоим алгоритмам совпадает
U=Z[Z["fault_RF"]==False] # оставляем только ошибочные по 1 алг. id
U=U[U["un"]==True] # получаем строки, в которых оба алгоритма ошиблись
picturing(U)