# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 14:40:48 2021

@author: Rhiznab
"""

# Cómo importar las librerias en Python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
# Importar el Data set
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values 
y = dataset.iloc[:, 2:].values 

from sklearn.linear_model import LinearRegression
regression_ln = LinearRegression()
regression_ln.fit(X, y)

# Regresión polinómica
# Incorporamos en el array las columnas de datos con los valores a distintos polinomios (cuadrado, al cubo, ...)
from sklearn.preprocessing import PolynomialFeatures
regression_poli = PolynomialFeatures(degree = 4)
X_poly = regression_poli.fit_transform(X)

# Se usa la misma LinearRegression pero con los datos en columnas con sus valores polinómicos
lin_reg_poly = LinearRegression() 
lin_reg_poly.fit(X_poly,y)

# Visualización del modelo lineal
plt.scatter(X, y, color="red")
plt.plot(X, regression_ln.predict(X),color="blue")
plt.title("Lineal")
plt.xlabel("Posicion Empleado")
plt.ylabel("Sueldo")
plt.show()


# Visualización del modelo polinómico
plt.scatter(X, y, color="red")
plt.plot(X, lin_reg_poly.predict(X_poly),color="blue")
plt.title("Polinomico")
plt.xlabel("Posicion Empleado")
plt.ylabel("Sueldo")
plt.show()

# Visualización del modelo polinómico. Ponemos más valores para una mejor gráfica y calculamos su predicción.
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X, y, color="red")
plt.plot(X_grid, lin_reg_poly.predict(regression_poli.fit_transform(X_grid)),color="blue")
plt.title("Polinomico")
plt.xlabel("Posicion Empleado")
plt.ylabel("Sueldo")
plt.show()

# Predicción de nuestros modelos
ln_pred = regression_ln.predict([[6.5]])

poly_pred = lin_reg_poly.predict(regression_poli.fit_transform([[6.5]]))
