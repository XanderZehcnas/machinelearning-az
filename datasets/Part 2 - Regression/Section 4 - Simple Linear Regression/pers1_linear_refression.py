# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 08:43:54 2021

@author: Rhiznab
"""

# Regresión Lineal simple
# y = b0 + b1*x

# Cómo importar las librerias en Python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
# Importar el Data set
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :-1].values 
y = dataset.iloc[:, 1].values 

# Dividir el data set en conjunto de entrenamiento y en conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 1/3, random_state = 0)

# Modelo de Regresión Lineal Simple con X_train e y_train
from sklearn.linear_model import LinearRegression

regression = LinearRegression()
regression.fit(X_train, y_train)

# Predecir el resultado del modelo de regresion entrenado
y_pred = regression.predict(X_test)


# Representación gráfica del resultado
plt.scatter(X_train,y_train, color="red")
plt.plot(X_train,regression.predict(X_train),color="blue")
plt.title("Sueldo vs Años de Experiencia")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo en $")
plt.show()

# Revisión de los puntos de Test
plt.scatter(X_test,y_test, color="red")
plt.plot(X_train,regression.predict(X_train),color="blue")
plt.title("Sueldo vs Años de Experiencia")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo en $")
plt.show()