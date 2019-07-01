from wsgiref import headers

import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np


data = pd.read_csv(r'C:\Users\Josh\Downloads\NBA_train.csv')
print(data)

# This tell us the numbers of missing value in each column and since all is zero,it shows us there is no missing value
data.isnull().sum()
print(data.isnull().sum())

# Dropping a feature not correlated to a target class so i drop the column of AST and STL
data_set = data.select_dtypes(include=[np.number]).interpolate().dropna()
X = data.drop(['AST','STL'], axis=1)
print(X)

#what we want to do is see what data types we have:
print(data.dtypes)

# Encoding the categorical feature, we are going to include only the object columns in our data frame.
obj_data = data.select_dtypes(include=['object']).copy()
print(obj_data)

# We use pandas to convert a column to a category, then use those category values for your label encoding
obj_data["Team"] = obj_data["Team"].astype('category')
print(obj_data.dtypes)

# Then we can assign the encoded variable to a new column using the cat.codes.
obj_data["Team_cat"] = obj_data["Team"].cat.codes
print(obj_data)