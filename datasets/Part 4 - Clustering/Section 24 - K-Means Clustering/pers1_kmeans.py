# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 13:30:56 2021

@author: Rhiznab
"""

# Clustering con KMeans
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Mall_Customers.csv')
# Los clientes tienen una valoración de lo que gastan y queremos saber como agrupar clientes por su edad, su sueldo y sus resultados finales. No usamos la edad en el ejemplo.

X = dataset.iloc[:,[3,4]].values

# Veamos la opción optmima de K
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init="k-means++", max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title("Método del codo")
plt.xlabel("NºClusters - k")
plt.ylabel("WCSS(k)")
plt.show()

# Elegimos k = 5 como numero optimo
kmeans = KMeans(n_clusters=5,init="k-means++", max_iter=300,n_init=10,random_state=0)
y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[y_kmeans == 0,0],X[y_kmeans == 0,1],s=100,c="red",label="Cluster 1")
plt.scatter(X[y_kmeans == 1,0],X[y_kmeans == 1,1],s=100,c="green",label="Cluster 2")
plt.scatter(X[y_kmeans == 2,0],X[y_kmeans == 2,1],s=100,c="yellow",label="Cluster 3")
plt.scatter(X[y_kmeans == 3,0],X[y_kmeans == 3,1],s=100,c="cyan",label="Cluster 4")
plt.scatter(X[y_kmeans == 4,0],X[y_kmeans == 4,1],s=100,c="magenta",label="Cluster 5")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c="black",label="Baricentros")
plt.title("K=5")
plt.xlabel("Sueldo (k$)")
plt.ylabel("Rate Level")
plt.legend()
plt.show()