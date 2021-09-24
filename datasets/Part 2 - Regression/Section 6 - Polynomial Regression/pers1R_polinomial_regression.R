# Regresión Polinómica

# Importar el dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[, 2:3]

# Modelo Regresión Lineal
# Viendo su summary se ve que hay numeros raros en el coeficiente cero y los incrementos lineales. No es lineal.
lin_reg = lm(formula = Salary ~ Level,data=dataset)


# Modelo Regresión Polinómico
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
pol_reg = lm(dataset$Salary ~ .,data=dataset)

# Visualización de los modelos

# Regresión Lineal
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),color = "red") + 
  geom_line(aes(x = dataset$Level, y = predict(lin_reg,newdata = dataset)),color = "blue") +
  ggtitle("Regresión Lineal") + 
  xlab("Level") + 
  ylab("Sueldo")
  

# Regresión polinómica
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),color = "red") + 
  geom_line(aes(x = dataset$Level, y = predict(pol_reg,newdata = dataset)),color = "blue") +
  ggtitle("Regresión Polinómica") + 
  xlab("Level") + 
  ylab("Sueldo")

# Regresión polinómica con suavizado de curva

X_grid = seq(min(dataset$Level),max(dataset$Level),0.1)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),color = "red") + 
  geom_line(aes(x = X_grid, y = predict(pol_reg,newdata = data.frame(Level = X_grid, Level2 = X_grid^2, Level3 = X_grid^3, Level4 = X_grid^4))),color = "blue") +
  ggtitle("Regresión Polinómica") + 
  xlab("Level") + 
  ylab("Sueldo")


# Predicción de nuevos resultados
x_check = data.frame(Level = 6.5)
y_pred = 0
y_pred = predict(lin_reg,newdata = x_check)

x_check = data.frame(Level = 6.5, Level2 = 6.5^2, Level3 = 6.5^3, Level4=6.5^4)
t_poly_pred = predict(pol_reg,newdata=x_check)
