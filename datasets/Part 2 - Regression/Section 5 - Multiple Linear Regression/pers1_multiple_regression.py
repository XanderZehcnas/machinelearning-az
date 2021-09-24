# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 17:27:26 2021

@author: Rhiznab
"""

# Cómo importar las librerias en Python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
# Regresion Lineal Multiple
# y = b0 + b1*x1 + b2*x2 + b3*x3 ...
""" 
Regresión Lineal debe cumplir:
- Linealidad
- Homocedasticidad
- Normalidad multivariable
- Independencia de errores
- Ausencia de multicolinealidad
"""

# Importar el Data set
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:, :-1].values 
y = dataset.iloc[:, 4].values 

# Codificar datos categoricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import make_column_transformer

labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = make_column_transformer((OneHotEncoder(), [3]), remainder = "passthrough")
X = onehotencoder.fit_transform(X)

# Evitar trampa de las variables Dummy
X = X[:,1:]

# Dividir el data set en conjunto de entrenamiento y en conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2, random_state = 0)

# Ajustar modelo de regresión lineal múltiple con el X_train.
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train,y_train)

# Prediccion de resultados en testing
y_pred = regression.predict(X_test)

# Construir el modelo óptimo de regresión lineal múltiple utiliizando la Eliminación hacia atrás
import statsmodels.api as sm

#Se agrega una columna de unos que será la que tome el coeficiente independiente para poder valorar su p-valor tambien
X = np.append(arr = np.ones((50,1)).astype(int), values = X , axis=1)

SL = 0.05

X_opt = X[:,[0,1,2,3,4,5]].tolist() # Empezamos con todas las columnas
regression_ols = sm.OLS(endog=y,exog = X_opt).fit()
regression_ols.summary()

# Quitamos ña variable x2
X_opt = X[:,[0,1,3,4,5]].tolist() # Empezamos con todas las columnas
regression_ols = sm.OLS(endog=y,exog = X_opt).fit()
regression_ols.summary()

# Quitamos variable
X_opt = X[:,[0,3,4,5]].tolist() # Empezamos con todas las columnas
regression_ols = sm.OLS(endog=y,exog = X_opt).fit()
regression_ols.summary()

# Quitamos variable
X_opt = X[:,[0,3,5]].tolist() # Empezamos con todas las columnas
regression_ols = sm.OLS(endog=y,exog = X_opt).fit()
regression_ols.summary()

# Quitamos variable
X_opt2 = X[:,[0,3]].tolist() # Empezamos con todas las columnas
regression_ols2 = sm.OLS(endog=y,exog = X_opt2).fit()
regression_ols2.summary()

# Elegiríamos 