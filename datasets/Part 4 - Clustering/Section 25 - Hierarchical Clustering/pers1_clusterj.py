# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 17:33:57 2021

@author: Rhiznab
"""

# Clustering jerárquico

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values


# Utilizar dendograma para encontrar el numero optimo de grupos
import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(X, method="ward"))
plt.title("Dendograma")
plt.xlabel("Clientes")
plt.ylabel("Distancia Euclídea")
plt.show()

# Vemos que 5 clusters son los óptimos. Al mirar las lineas verticales hay que contar sólo el trozo que no cruza lineas horizontales de otros grupos.
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity="euclidean",linkage="ward")

y_hc = hc.fit_predict(X)

# Visualización de los lcusters

plt.scatter(X[y_hc == 0,0],X[y_hc == 0,1],s=100,c="red",label="Cluster 1")
plt.scatter(X[y_hc == 1,0],X[y_hc == 1,1],s=100,c="green",label="Cluster 2")
plt.scatter(X[y_hc == 2,0],X[y_hc == 2,1],s=100,c="yellow",label="Cluster 3")
plt.scatter(X[y_hc == 3,0],X[y_hc == 3,1],s=100,c="cyan",label="Cluster 4")
plt.scatter(X[y_hc == 4,0],X[y_hc == 4,1],s=100,c="magenta",label="Cluster 5")
plt.title("Cluster Jerárquico K=5")
plt.xlabel("Sueldo (k$)")
plt.ylabel("Rate Level")
plt.legend()
plt.show()