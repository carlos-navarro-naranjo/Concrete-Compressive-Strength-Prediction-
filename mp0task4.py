# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 15:48:18 2022

@author: cnava
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV






def cleanse_data  (data):
 
 data.dropna()
 
 data= data[(data >= 0).all(axis=1)]


 return data

df = cleanse_data(pd.read_csv("concrete_data_stu.csv"))


def eval_model (pipe, X_train, y_train, X_test, y_test, name):
     d = dict();
     d['r2 score'] = r2_score(y_test, pipe.predict(X_test))
     d['rmse test']   = mean_squared_error(y_test, pipe.predict(X_test))
     d['rmse train']   = mean_squared_error(y_train, pipe.predict(X_train))
     d['name']   = name
     return d

features = ['Cement', 
            'Slag', 
            'Fly Ash', 'Water', 
            'SP', 
            'CourseAgg', 
            'FineAgg', 
            'Age'] 


response = 'Strength'

X = df[features]
y = df[response]



X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size=0.8, 
                                                    random_state=42)




# create pipeline
estimators = [('scaler', StandardScaler()),
              ('svr', SVR(kernel='rbf'))]
pipe = Pipeline(estimators)

# define and execute grid search
params = {'svr__C' : 10.0 ** np.arange(-2,2), 'svr__epsilon' : np.logspace(-1,1, num=5), 'svr__gamma' : np.logspace(-3,1, num=5)}
clfs = GridSearchCV( Pipeline(estimators), params, cv=8) 
clfs.fit(X_train, y_train)
print('Best parameter C: ', clfs.best_params_ ['svr__C'])
print('Best parameter epsilon: ', clfs.best_params_['svr__epsilon'])
print('Best parameter gamma: ', clfs.best_params_ ['svr__gamma'])
clfs=clfs.best_estimator_
results= eval_model (clfs, X_train, y_train, X_test, y_test, name='SVR: cv=8 ')
print('RMSE TEST: ', '{0:.6g}'.format(results['rmse test']))
print('RMSE TRAIN:','{0:.6g}'.format( results['rmse train']))


