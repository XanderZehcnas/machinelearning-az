# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 13:32:42 2021

@author: Rhiznab
"""
# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)

transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])

# Entrnar el algoritmo de apriori
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence =0.2, min_lift=3, min_length=2)

results = list(rules)
print(results[0])


