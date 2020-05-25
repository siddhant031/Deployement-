#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 15:29:03 2020

@author: siddhant
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

dataset = pd.read_csv("Employee_Salary copy.csv")  
X= dataset.iloc[:, :3]
y=dataset.iloc[:, -1]

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X,y)
pickle.dump(regressor, open('model.pkl','wb'))

model= pickle.load(open('model.pkl','rb'))
 
print(model.predict([[2,9,6]]))
