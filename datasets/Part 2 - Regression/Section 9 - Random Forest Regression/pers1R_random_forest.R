
# Bosques Aleatorios

# Importar el dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[, 2:3]

# Ajustar Modelo de RegresiÃ³n con el Conjunto de Datos
library(randomForest)
regression = randomForest(x = dataset[1], y = dataset$Salary, ntree=100)

# PredicciÃ³n de nuevos resultados con RegresiÃ³n 
y_pred = predict(regression, newdata = data.frame(Level = 6.5))



# VisualizaciÃ³n del modelo de regresiÃ³n
# install.packages("ggplot2")
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level , y = dataset$Salary),
             color = "red") +
  geom_line(aes(x = x_grid, y = predict(regression, 
                                        newdata = data.frame(Level = x_grid))),
            color = "blue") +
  ggtitle("PredicciÃ³n (Modelo de RegresiÃ³n)") +
  xlab("Nivel del empleado") +
  ylab("Sueldo (en $)")