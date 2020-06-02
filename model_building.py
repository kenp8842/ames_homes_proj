# -*- coding: utf-8 -*-
"""
Created on Wed May 27 15:37:01 2020

@author: kenp8
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


df = pd.read_csv('C:/Users/kenp8/Documents/ames_homes_proj/ames_homes_transformed.csv')

#choose relevant columns
df.columns

df_model = df[['SalePrice', 'MS SubClass', 'Neighborhood', 'House Style', 'Overall Qual', 'Garage Area', 'Misc Val',
               'liveable_sf', 'bath_total', 'effective_age']]

#get dummy data
df_dum = pd.get_dummies(df_model)

#train test split
from sklearn.model_selection import train_test_split

X = df_dum.drop('SalePrice', axis=1)
y = df_dum['SalePrice'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=64)

#linear regression
#using stats model to get idea what factors influence model the most
import statsmodels.api as sm
X_sm = sm.add_constant(X)
model = sm.OLS(y, X_sm)
print(model.fit().summary())

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score

lm = LinearRegression()
lm.fit(X_train, y_train)
print('OLS:', np.mean(cross_val_score(lm, X_train, y_train, scoring ='neg_mean_absolute_error', cv=3)))

#lasso regression
lm_l = Lasso()
lm_l.fit(X_train, y_train)
print('Lasso default:', np.mean(cross_val_score(lm_l, X_train, y_train, scoring ='neg_mean_absolute_error', cv=3)))

alpha = []
error = []

for i in range(1, 30):
    alpha.append(i)
    lml = Lasso(alpha = i)
    error.append(np.mean(cross_val_score(lml, X_train, y_train, scoring ='neg_mean_absolute_error', cv=3)))
    
plt.plot(alpha, error)   

err = tuple(zip(alpha, error))  
df_err = pd.DataFrame(err, columns = ['alpha', 'error'])
df_err[df_err.error == max(df_err.error)]
print('Lasso alpha optimized:', max(error))
    
#random forest 
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=64)

print('Random Forest default:', np.mean(cross_val_score(rf, X_train, y_train, scoring ='neg_mean_absolute_error', cv=3)))

#tune models using GridSearchCV
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':range(10, 300, 10), 'criterion':('mse', 'mae'), 'max_features': ('auto', 'sqrt', 'log2')} 

gs=GridSearchCV(rf, parameters, scoring='neg_mean_absolute_error', cv=3)
gs.fit(X_train, y_train)

print('\n', 'GridSearchCV:', gs.best_score_, 'GridSearchCV parameters:', gs.best_estimator_)

#test ensembles 
tpred_lm = lm.predict(X_test)
tpred_lml = lm_l.predict(X_test)
tpred_rf = gs.best_estimator_.predict(X_test)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, tpred_lm)
mean_absolute_error(y_test, tpred_lml)
mean_absolute_error(y_test, tpred_rf) 

 