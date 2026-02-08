Eg: Grouping similar news, Market segmentation 

1. K-means clustering 

Step 0: Random initialization on centroids( always less than number of training examples)
Step 1: Assign points to cluster centroids
Step 2: Move cluster centroids

If cluster has 0 members either eliminate it or randomly initialize it again.

=====================================================================================================

Optimization objective 

=====================================================================================================

Random initialization 

- Run k means multiple times using different random initialization 
- Find the least value of cost function to select best clustering

====================================================================================================

Choosing the number of clusters

- Elbow method
- Based on downstream usecase of the model