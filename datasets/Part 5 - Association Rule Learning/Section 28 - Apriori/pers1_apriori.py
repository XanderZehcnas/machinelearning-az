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
rules = apriori(transactions, min_support = )



# ------------------------------------------------------------------------
# GOAL: show how to create html widgets with transaction rules
# ------------------------------------------------------------------------
 
# libraries --------------------------------------------------------------
library(arules)
library(arulesViz)
 
# data -------------------------------------------------------------------
path <- "~/Downloads/P14-Part5-Association-Rule-Learning/Section 28 - Apriori/"
trans <- read.transactions(
file = paste0(path, "R/Market_Basket_Optimisation.csv"),
sep = ",",
rm.duplicates = TRUE
)
 
# apriori algoirthm ------------------------------------------------------
rules <- apriori(
data = trans,
parameter = list(support = 0.004, confidence = 0.2)
)
 
# visualizations ---------------------------------------------------------
plot(rules, method = "graph", engine = "htmlwidget")