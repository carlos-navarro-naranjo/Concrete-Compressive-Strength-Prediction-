# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 19:50:58 2022

@author: cnava
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor







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



##Linear Regression Technique

poly3 = PolynomialFeatures(degree=3)
model3 =[('linear Regression', LinearRegression(fit_intercept=False))]
pipe= Pipeline(model3)
pipe.fit(poly3.fit_transform(X_train), y_train)
#pipe.transform(X_train)
#pipe.transform(X_test)
results= eval_model (pipe, poly3.transform(X_train), y_train, poly3.transform(X_test), y_test, name='Linear Regression: 3rd Degree Polynomial')
print(results)
tuple_poly3=(pipe, results)
import pickle
pickle.dump( tuple_poly3, open('poly3.dat', 'wb'))




##ANN Technique 1

scaler=StandardScaler()
mlp=MLPRegressor(activation='relu', solver='sgd', hidden_layer_sizes=(100), max_iter=2500, random_state=42)
pipe=Pipeline([('estimator',mlp)])
pipe.fit(scaler.fit_transform(X_train),y_train)
results= eval_model (pipe, scaler.fit_transform(X_train), y_train, scaler.fit_transform(X_test), y_test, name='ANN: One hidden layer of 100 nodes, relu activation function and sequential gradient descent solver, 2500 max iterations ')
print(results)
tuple_ANN=(pipe, results)
import pickle
pickle.dump( tuple_ANN, open('ann1.dat', 'wb'))



##ANN Technique 2

scaler=StandardScaler()
mlp=MLPRegressor(activation='tanh', solver='sgd', hidden_layer_sizes=(100), max_iter=2500, random_state=42)
pipe=Pipeline([('estimator',mlp)])
pipe.fit(scaler.fit_transform(X_train),y_train)
results= eval_model (pipe, scaler.fit_transform(X_train), y_train, scaler.fit_transform(X_test), y_test, name='ANN: One hidden layer of 100 nodes, relu activation function and sequential gradient descent solver, 2500 max iterations ')
print(results)
tuple_ANN=(pipe, results)
import pickle
pickle.dump( tuple_ANN, open('ann2.dat', 'wb'))



##SVR

est = [('scaler', StandardScaler()),
       ('svr', SVR(epsilon=5, kernel='rbf', gamma=0.1, C=10))]
pipe = Pipeline(est)
pipe.fit(X_train, y_train)
results= eval_model (pipe, X_train, y_train, X_test, y_test, name='SVR: rbf kernel, epsilon of 5, C=10, gamma=0.1 ')
print(results)
tuple_ANN=(pipe, results)
import pickle
pickle.dump( tuple_ANN, open('svr.dat', 'wb'))


##Random Forest Technique

est = [('RandomForest', RandomForestRegressor(n_estimators=100))]
pipe = Pipeline(est)
pipe.fit(X_train, y_train)
results= eval_model (pipe, X_train, y_train, X_test, y_test, name='Random Forest: 100 trees ')
print(results)
tuple_ANN=(pipe, results)
import pickle
pickle.dump( tuple_ANN, open('rfr.dat', 'wb'))

##ADABOOST Reg Technique

est = [('ADABOOST', AdaBoostRegressor(n_estimators=100, loss='square'))]
pipe = Pipeline(est)
pipe.fit(X_train, y_train)
results= eval_model (pipe, X_train, y_train, X_test, y_test, name='ADABOOST: 100 trees stumps, loss function= square')
print(results)
tuple_ANN=(pipe, results)
import pickle
pickle.dump( tuple_ANN, open('ada.dat', 'wb'))


