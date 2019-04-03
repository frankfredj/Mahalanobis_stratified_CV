# Input

* data: a pandas data frame, containing numerical or categorical predictors
* y: the (numerical) target variable
* pca_var: the % of variance to be captured by PCA
* nfolds: the number of folds to be returned

# Methodology

The data is converted to a 2D numerical numpy array, and the scaled / centered PC matrix is obtained. K-means is applied to the 
PC matrix in a way such that the minimum cluster size is greater than or equal to int(len(y) / nfolds). In case of ties, silhouette
scores are used to determine the optimal number of centroids. Quantile-based stratified sampling is then applied to the target
variable for each computed clusters, and folds are iteratively merged together. Finally, the respective coefficients of variation of the
(per folds) 1st, 2nd, 3rd and 4th central moments are computed and printed.
