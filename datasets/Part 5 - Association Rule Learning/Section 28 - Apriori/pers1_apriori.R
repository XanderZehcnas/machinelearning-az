# Reglas de asociación de Apriori en R

dataset_orig = read.csv('Market_Basket_Optimisation.csv',header=FALSE)

# Preprocesado de datos ... Tendremos una matriz Sparse con muchos ceros, dado que el dataset tiene mucho vacío
library(arules)
dataset = read.transactions('Market_Basket_Optimisation.csv', sep=",", rm.duplicates = TRUE)

itemFrequencyPlot(dataset,topN=20)

# Entrenar algoritmo de Apriori con el dataset
rules = apriori(data = dataset, 
        parameter = list(support = 0.01, confidence=0.2))

# Visualización de los resultados
inspect(sort(rules, by='lift')[1:10])


# ------------------------------------------------------------------------
# GOAL: show how to create html widgets with transaction rules
# ------------------------------------------------------------------------

# libraries --------------------------------------------------------------
library(arules)
library(arulesViz)

# data -------------------------------------------------------------------
path <- "D:/Desarrollo/GitHubForks/machinelearning-az/datasets/Part 5 - Association Rule Learning/Section 28 - Apriori/"
trans <- read.transactions(
  file = paste0(path, "Market_Basket_Optimisation.csv"),
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