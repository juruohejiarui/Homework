from sklearn.cluster import KMeans
from sklearn.metrics import fowlkes_mallows_score, rand_score, adjusted_rand_score, v_measure_score, \
	silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
import data
import visualize
import matplotlib.pyplot as plt

def kmeans(X : np.ndarray, n_clusters: int, n_init: int = 10) -> tuple :
	km = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=600)

	km.fit(X)

	return km.labels_, km.cluster_centers_

def hierarchical(X : np.ndarray, n_clusters: int) -> tuple :
	from sklearn.cluster import AgglomerativeClustering

	ag = AgglomerativeClustering(n_clusters=n_clusters)

	ag.fit(X)

	return ag.labels_, ag.children_

def metric(name : str, X : np.ndarray, y : np.ndarray, labels : np.ndarray, centers : np.ndarray) :
	print(f"metric of {name}")

	print("external metric: ")

	# fowlkes_mallows score
	print(f"\tfowlkes_mallows score : {fowlkes_mallows_score(y, labels)}")

	# rand score
	print(f"\trand score : {rand_score(y, labels)}")

	# adjusted rand score
	print(f"\tadjusted rand score : {adjusted_rand_score(y, labels)}")

	# v measure score
	print(f"\tv measure score : {v_measure_score(y, labels)}")

	print("internal metric: ")

	# silhouette score
	print(f"\tsilhouette score : {silhouette_score(X, labels)}")

	# calinski harabasz score
	print(f"\tcalinski harabasz score : {calinski_harabasz_score(X, labels)}")
	# davies bouldin score
	print(f"\tdavies bouldin score : {davies_bouldin_score(X, labels)}")

	print()



if __name__ == "__main__" :
	X, y = data.load_data("./Data/train", "train")

	x_dem = visualize.dimension_reduction(X, 2, "TSNE")

	labels_kmeans_dem, centers_kmeans_dem = kmeans(x_dem, 6, 20)
	# cluster using the orginal data
	labels_kmeans_org, centers_kmeans_org = kmeans(X, 6, 20)

	labels_hierarchical_dem, children_hierarchical_dem = hierarchical(x_dem, 6)
	# cluster using the orginal data
	labels_hierarchical_org, children_hierarchical_org = hierarchical(X, 6)
	
	metric("kmeans orginal", X, y, labels_kmeans_org, centers_kmeans_org)
	metric("kmeans dimension reduction", x_dem, y, labels_kmeans_dem, centers_kmeans_dem)

	metric("hierarchical orginal", X, y, labels_hierarchical_org, children_hierarchical_org)
	metric("hierarchical dimension reduction", x_dem, y, labels_hierarchical_dem, children_hierarchical_dem)

	# plot the clusters
	visualize.plot(
		[x_dem, x_dem, x_dem, x_dem], 
		[labels_kmeans_org, labels_kmeans_dem, labels_hierarchical_org, labels_hierarchical_dem], 
		2, False, ["kmean org", "kmean dem", "hier org", "hier dem"], "cluster.png")