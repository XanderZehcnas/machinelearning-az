# Clustering con K-Means

dataset = read.csv('Mall_Customers.csv')
X = dataset[, 4:5]

# KMeans - Veamos el numero optimo de K con la técnica del codo
set.seed(6)
wcss = vector()

for (i in 1:10) {
  wcss[i] <- sum(kmeans(X,i)$withinss)
}
plot(1:10,wcss, type="b", main="Metodo del codo", xlab="Clusters (k)", ylab="WCSS(k)")


# Elegimos el K = 5 o k=6
set.seed(29)
kmeans_res <- kmeans(X,5,iter.max=300,nstart=10)

# Visualización de clusters
library(cluster)
clusplot(X,
         kmeans_res$cluster,
         lines=0,
         shade=TRUE,
         color=TRUE,
         labels=2,
         plotchar=FALSE,
         span=TRUE,
         main="Clustering de clientes",
         xlab="Ingresos anuales",
         ylab="Score (1-100)")


