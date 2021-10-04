# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 13:27:51 2021

@author: Rhiznab
"""

# Natural Language Processing (NLP) - Procesamiento de lenguaje natural

# Importar librerias básicas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset. Delimitador tabulador. Ignorar comillas dobles
dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter="\t", quoting=3)

# Limpieza de los datos y paso a minusculas para evitar que luego tengamos demasiadas palabras

import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0,len(dataset)):
    review = re.sub('[^a-zA-Z]',' ',dataset["Review"][i]) # Nos quedamos con las letras de la a a la Z y de la A a la Z
    review = review.lower() # A minusculas
    review = review.split()
    
    # Quitar palabras no útiles para valorar
    # Palabras conjugadas, en plural, etc. se traducen a la raiz de la palabra (PorterStemmer)
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    
    corpus.append(' '.join(review))


# Crear a bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500) # Max features para quitar las palabras que aparecen poco, se podría calcular cuales aparecen poco para quitar más
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1]

# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

## Kernel SVM
##--------------------------------------
# Ajustar el clasificador en el Conjunto de Entrenamiento
from sklearn.svm import SVC
classifier = SVC(kernel = "rbf", random_state = 0)
classifier.fit(X_train, y_train)

# Predicción de los resultados con el Conjunto de Testing
y_pred  = classifier.predict(X_test)

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm_ksvm = confusion_matrix(y_test, y_pred)
acc_ksvm = (cm_ksvm[0][0] + cm_ksvm[1][1]) / len(y_test)
pre_ksvm = cm_ksvm[1][1] / (cm_ksvm[1][1] + cm_ksvm[0][1])
recall_ksvm = cm_ksvm[1][1] / (cm_ksvm[1][1] + cm_ksvm[1][0])
score_ksvm = 2*pre_ksvm*recall_ksvm/(pre_ksvm+recall_ksvm)


#SVM 
##--------------------------------------
# Ajustar el clasificador en el Conjunto de Entrenamiento
from sklearn.svm import SVC
classifier = SVC(kernel="linear",random_state=0)
classifier.fit(X_train,y_train)


# Predicción de los resultados con el Conjunto de Testing
y_pred  = classifier.predict(X_test)

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm_svm = confusion_matrix(y_test, y_pred)
acc_svm = (cm_svm[0][0] + cm_svm[1][1]) / len(y_test)
pre_svm = cm_svm[1][1] / (cm_svm[1][1] + cm_svm[0][1])
recall_svm = cm_svm[1][1] / (cm_svm[1][1] + cm_svm[1][0])
score_svm = 2*pre_svm*recall_svm/(pre_svm+recall_svm)

## Naive Bayes
##--------------------------------------
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

# Predicción de los resultados con el Conjunto de Testing
y_pred  = classifier.predict(X_test)

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm_nb = confusion_matrix(y_test, y_pred)
acc_nb = (cm_nb[0][0] + cm_nb[1][1]) / len(y_test)
pre_nb = cm_nb[1][1] / (cm_nb[1][1] + cm_nb[0][1])
recall_nb = cm_nb[1][1] / (cm_nb[1][1] + cm_nb[1][0])
score_nb = 2*pre_nb*recall_nb/(pre_nb+recall_nb)

# Losgistic Regression
##--------------------------------------
# Ajustar el modelo de regresión logística con el dataset
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)

classifier.fit(X_train, y_train)

# Predicción de los resultados
y_pred = classifier.predict(X_test)

# Verificar el rendimiento con una matriz de confusion
from sklearn.metrics import confusion_matrix

cm_log = confusion_matrix(y_test, y_pred)
acc_log = (cm_log[0][0] + cm_log[1][1]) / len(y_test)
pre_log = cm_log[1][1] / (cm_log[1][1] + cm_log[0][1])
recall_log = cm_log[1][1] / (cm_log[1][1] + cm_log[1][0])
score_log = 2*pre_log*recall_log/(pre_log+recall_log)

# K-Nearest Neighbors
##--------------------------------------
# Ajustar el clasificador en el Conjunto de Entrenamiento - KNN
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5,metric="minkowski",p=2) #p=2 para medidas euclideas
classifier.fit(X_train,y_train)


# Predicción de los resultados con el Conjunto de Testing
y_pred  = classifier.predict(X_test)

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm_knn = confusion_matrix(y_test, y_pred)
acc_knn = (cm_knn[0][0] + cm_knn[1][1]) / len(y_test)
pre_knn = cm_knn[1][1] / (cm_knn[1][1] + cm_knn[0][1])
recall_knn = cm_knn[1][1] / (cm_knn[1][1] + cm_knn[1][0])
score_knn = 2*pre_knn*recall_knn/(pre_knn+recall_knn)

# Decision Tree
##--------------------------------------

# Ajustar el clasificador en el Conjunto de Entrenamiento
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion="entropy",random_state=0)
classifier.fit(X_train,y_train)


# Predicción de los resultados con el Conjunto de Testing
y_pred  = classifier.predict(X_test)

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm_tree = confusion_matrix(y_test, y_pred)
acc_tree = (cm_tree[0][0] + cm_tree[1][1]) / len(y_test)
pre_tree = cm_tree[1][1] / (cm_tree[1][1] + cm_tree[0][1])
recall_tree = cm_tree[1][1] / (cm_tree[1][1] + cm_tree[1][0])
score_tree = 2*pre_tree*recall_tree/(pre_tree+recall_tree)

## Random Forest
##--------------------------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=1000, criterion="entropy", random_state=0)
classifier.fit(X_train,y_train)
    
y_pred  = classifier.predict(X_test)

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm_rfor = confusion_matrix(y_test, y_pred)
acc_rfor = (cm_rfor[0][0] + cm_rfor[1][1]) / len(y_test)
pre_rfor = cm_rfor[1][1] / (cm_rfor[1][1] + cm_rfor[0][1])
recall_rfor = cm_rfor[1][1] / (cm_rfor[1][1] + cm_rfor[1][0])
score_rfor = 2*pre_rfor*recall_rfor/(pre_rfor+recall_rfor)

# c5.0
import c50
