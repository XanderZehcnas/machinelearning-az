# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 16:32:10 2021

@author: Rhiznab
"""

# Plantilla de preprocesado

# Importar las librerias
print("Carga de librerías")
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:3].values
y = dataset.iloc[:,3].values

# Tratamiento de los NAs
from sklearn.impute import SimpleImputer

SImputer = SimpleImputer(strategy="mean")
X[:,1:3] = SImputer.fit_transform(X[:,1:3])

# Codificar datos categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# The column numbers to be transformed (here is [0] but can be [0, 1, 3])
# Leave the rest of the columns untouched
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],remainder='passthrough') 

X = np.array(ct.fit_transform(X), dtype=np.float)
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Dividir dataset en conjunto de entrenamiento y testing
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2, random_state=0)

# Escalado de variables
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Aqui ha escalado tambien las columnas categoricas que s epodría no haber hecho