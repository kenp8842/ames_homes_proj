# -*- coding: utf-8 -*-
"""
Created on Thu May 21 14:58:19 2020

@author: kenp8
"""

import pandas as pd
pd.options.display.max_columns = 999
import numpy as np

df = pd.read_csv(r"C:\Users\kenp8\Downloads\AmesHousing.txt", delimiter = '\t')
#df.info()

#drop columns with little info., 
#fill-in NaN values

#drop columns with little information
##total missing by column
num_missing = df.isnull().sum()
print(num_missing)
 
##filter columns >40% missing values
drop_miss_cols = num_missing[(num_missing > (df.shape[0] * .4 ))].sort_values()

##drop these columns from dataframe
df = df.drop(drop_miss_cols.index, axis=1)
#print(df.shape)

#filling in missing values
fillable_cols = num_missing[num_missing > 0].sort_values()

##fill in missing electrical with most common
df['Electrical'] = df['Electrical'].fillna('SBrkr')


##Mas Vnr Type fill with most common
#print(df['Mas Vnr Type'].value_counts(dropna=False))
df['Mas Vnr Type'] = df['Mas Vnr Type'].fillna('None')
#print(df['Mas Vnr Type'].isnull().sum())

#Basement/garage columns
##basvalue_counts(dropna=False))
base_grg_cols = ['Bsmt Exposure', 'BsmtFin Type 2', 'Bsmt Qual', 'Bsmt Cond', 'BsmtFin Type 1',
                 'Garage Type', 'Garage Finish', 'Garage Qual', 'Garage Cond']
for col in base_grg_cols:
    df[col] = df[col].fillna('None')

#getting year if no garage = 0 so it stays a float    
df.loc[df['Garage Type'] == 'None','Garage Yr Blt'] = 0
  
#print(df[base_grg_cols].isnull().sum())
    
    
##Fil in NaN in numeric columns
#find numeric columns number of missing values
num_col_missing = df.select_dtypes(include =['integer', 'float']).isnull().sum()

#identifying columns with missing numerical values to fill in
fillable_numeric_cols = num_col_missing[(num_col_missing < df.shape[0]/2) &  (num_col_missing > 0)].sort_values()

#find most common values in numerical columns with missing values
num_replace_dict = df[fillable_numeric_cols.index].mode().to_dict(orient='records')[0]

#use dictionary to fill in most common values in numeric columns
df = df.fillna(num_replace_dict)

#dropping unnecesary first column
df_cleaned = df.drop('Order', axis=1)
