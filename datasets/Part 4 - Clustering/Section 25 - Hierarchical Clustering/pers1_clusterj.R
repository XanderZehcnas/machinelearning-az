# Clustering Jerarquico

dataset = read.csv('Mall_Customers.csv')
X = dataset[, 4:5]

# Utilizar el dendrograma
dendrogram = hclust(dist(X,method="euclidean"),
                    method="ward.D")
plot(dendrogram,
     main="Dendrograma",
     xlab="Clientes Centro Comercial",
     ylab="Distancia euclidea")


# Ajustar clustering jerárquico - Elegimos el K = 5

hc = hclust(dist(X,method="euclidean"),
            method="ward.D")
y_hc = cutree(hc,k=5)


# VisualizaciÃ³n de clusters
library(cluster)
clusplot(X,
         y_hc,
         lines=0,
         shade=TRUE,
         color=TRUE,
         labels=2,
         plotchar=FALSE,
         span=TRUE,
         main="Clustering de clientes con cluster jerárquico",
         xlab="Ingresos anuales",
         ylab="Score (1-100)")
