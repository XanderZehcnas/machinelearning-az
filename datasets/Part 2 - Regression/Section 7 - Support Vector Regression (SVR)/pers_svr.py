# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 17:27:07 2021

@author: Rhiznab
"""


# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Escalado de Variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_Y.fit_transform(y.reshape(-1,1))

# Modelo de Regresión SVR
from sklearn.svm import SVR
regression = SVR(kernel="rbf")
regression.fit(X,y)

# Prediccion
y_pred = regression.predict(sc_X.transform([[6.5]]))
y_pred = sc_Y.inverse_transform(y_pred)


# Visualización de resultados SVR
X_grid = np.arange(min(X), max(X), 0.02)
X_grid = X_grid.reshape(len(X_grid), 1)

X_grid_show = sc_X.inverse_transform(X_grid)

y_pred_grid = sc_Y.inverse_transform(regression.predict(X_grid))

orig_X = sc_X.inverse_transform(X)
orig_Y = sc_Y.inverse_transform(y)

plt.scatter(orig_X, orig_Y, color = "red")
plt.plot(X_grid_show, y_pred_grid, color = "blue")
plt.title("Modelo de Regresión")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()
