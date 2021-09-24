
# Importar el dataset
dataset = read.csv('50_Startups.csv')

# Codificar variables categóricas
dataset$State = factor(dataset$State,levels=c("New York","Florida","California"),labels=c(1,2,3))

# Dividir en conjunto de test y entrenamiento
library(caTools)
split = sample.split(dataset$Profit, SplitRatio=0.8)

training_set = subset(dataset,split == TRUE)
testing_set = subset(dataset,split == FALSE)

# Ajustar el modelo de regresión lineal múltiple con el Conjunto de Entrenamiento
# El punto es para indicar en la fórmula que todas las otras variables
# R automaticamente descarta uno de los factores dado que sabe que ponerlo sería entrar en multicolinealidad
regression = lm(formula = Profit ~ ., data = training_set)

# Predecir los resultados con el conjunto de testing
y_pred = predict(regression,testing_set)

# Construir modelo optimo con la Eliminación hacia atrás (usamos las columnas para poder ir quitando de normal habria que buscar algo más automatico)
SL = 0.05
regression = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State, data = dataset)
summary(regression)

regression = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend, data = dataset)
summary(regression)

regression = lm(formula = Profit ~ R.D.Spend + Marketing.Spend, data = dataset)
summary(regression)

regression = lm(formula = Profit ~ R.D.Spend, data = dataset)
summary(regression)