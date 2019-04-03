import pandas as pd 

data = pd.read_csv(r"C:\Users\Francis\Documents\KidneyTest2.csv")






def Mahalanobis_stratified_CV(data, y, pca_var, nfolds):


	import numpy as np
	y = np.array(y)

	#Get numerical and categorical column indices

	def IsNumeric (data):

		data_types = data.dtypes

		index = 0

		for i in data_types:

			if(i == "int64" or i == "float64"):
				data_types[index] = True

			else:
				data_types[index] = False

			index += 1

		return data_types



	#Obtain a centered / scaled matrix with numerical values
	#n factor levels -> (0, ... , n-1)

	numerical_data = np.empty(shape = data.shape)

	data_type_index = IsNumeric(data)


	for i in range(0,data.shape[1]):

		if data_type_index[i]:

			numerical_data[:,i] = data.iloc[:,i]


		else:

			levels = data.iloc[:,i].unique()

			dummy_value = 0

			for j in levels:

				index = np.where(data.iloc[:,i] == j)

				numerical_data[index,i] = dummy_value

				dummy_value += 1

	numerical_data = (numerical_data - np.mean(numerical_data, axis = 0)) / np.std(numerical_data, axis = 0)


	#PCA decomposition

	from sklearn.decomposition import PCA

	pca = PCA(n_components=pca_var, svd_solver='full')

	numerical_data_PCA = pca.fit_transform(numerical_data)

	numerical_data_PCA = (numerical_data_PCA - np.mean(numerical_data_PCA, axis = 0)) / np.std(numerical_data_PCA, axis = 0)



	from sklearn.cluster import KMeans
	from multiprocessing.dummy import Pool
	import functools
	import multiprocessing


	#Chooses the best number of clusers based on the best silhouette score
	#minimum group size of cluster >= nfolds
	#best silhouette score 

	kmeans = KMeans()

	clusters = kmeans.fit_predict(numerical_data)

	ncore = max(multiprocessing.cpu_count() - 2, 1)


	def min_cluster_size(n_clusters, kmeans, X):

		from sklearn.cluster import KMeans
		from sklearn.metrics import silhouette_score

		kmeans.set_params(n_clusters=n_clusters, n_jobs = ncore, n_init = 30)
		clusters = kmeans.fit_predict(X)
		score = silhouette_score(X, clusters)

		ids = np.unique(clusters)

		min_size = np.empty(shape = len(ids))

		count = 0

		for i in ids:
			min_size[count] = np.shape(np.where(clusters == i))[1]
			count += 1

		min_size = min_size.astype(int)

		return [score, np.min(min_size)]


	results = np.empty(shape = (nfolds, 2))

	for i in range(2, nfolds+2):
		results[(i-2),:] = min_cluster_size(i, kmeans, numerical_data_PCA)

	treshold = int(len(y) / nfolds)

	accepted_n_clusters = np.where(results[:,1] >= treshold)[0]

	if len(accepted_n_clusters) == 0:

		accepted_n_clusters = np.array(list(range(2, nfolds + 2)))


	best_accepted_n_clusters = int(accepted_n_clusters[np.argsort(-results[accepted_n_clusters,0])[0]]) + 2


	kmeans.set_params(n_clusters=best_accepted_n_clusters, n_jobs = ncore)
	clusters = kmeans.fit_predict(numerical_data_PCA)


	#Quantile-based k-fold stratification

	def stratified_sample(y, nfolds, index):

		y = np.array(y[index])

		n = max(int(len(y) / nfolds), 1)

		q = np.linspace(0, 1, num = (n+1))
		q = np.quantile(y, q)
		q[0] = q[0] - 1
		q[len(q)-1] = q[len(q)-1] + 1

		out = [[] for i in range(nfolds)]


		for i in range(0,n):

			index_temp = np.where((y >= q[i]) & (y < q[i+1]))[0]
			index_temp = index[0][index_temp]
			np.random.shuffle(index_temp)
			folds = np.array_split(index_temp, nfolds)

			for j in range(0, nfolds):

				out[j].append(folds[j])


		for i in range(0,nfolds):

			out[i] = np.concatenate(out[i])


		return out

	#Fuse two list of indices together

	def fuse(f1, f2):

		f3 = []

		for i in range(0, len(f1)):

			f3.append(np.concatenate([f1[i], f2[i]]))

		return np.array(f3)


	#Initiate fold list, then fill it sequentially by looping through clusters

	folds = np.array([[] for i in range(0,nfolds)])


	for i in range(0, best_accepted_n_clusters):

		index = np.where(clusters == i)
		folds = fuse(folds, stratified_sample(y, nfolds, index))
		np.random.shuffle(folds)


	for i in range(0, nfolds):
		folds[i] = folds[i].astype(int)


	return folds















