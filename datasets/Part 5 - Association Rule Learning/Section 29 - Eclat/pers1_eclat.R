# Asociación con Eclat

dataset_orig = read.csv('Market_Basket_Optimisation.csv',header=FALSE)

library(arules)
dataset = read.transactions('Market_Basket_Optimisation.csv', sep=",", rm.duplicates = TRUE)

itemFrequencyPlot(dataset,topN=20)

rules = eclat(data = dataset, parameter = list(support=0.004, minlen = 2))

inspect(sort(rules, by='support')[1:10])
