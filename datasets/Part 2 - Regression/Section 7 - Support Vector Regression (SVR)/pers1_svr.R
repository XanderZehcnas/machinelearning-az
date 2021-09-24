# SVR

# Importar el dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[, 2:3]

# Escalado
# training_set[,2:3] = scale(training_set[,2:3])
# testing_set[,2:3] = scale(testing_set[,2:3])


# Ajustar el modelo de regresion
library(e1071)
regression = svm(formula = Salary ~ ., data = dataset, type="eps-regression", kernel="radial")


# Prediccion de nuevos resultados con Regresión 
y_pred = predict(regression, newdata = data.frame(Level = 6.5))



# Visualizacion del modelo de regresión
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level , y = dataset$Salary),
             color = "red") +
  geom_line(aes(x = x_grid, y = predict(regression, 
                                        newdata = data.frame(Level = x_grid))),
            color = "blue") +
  ggtitle("Prediccion (Modelo de Regresion con SVM)") +
  xlab("Nivel del empleado") +
  ylab("Sueldo (en $)")
