from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, fowlkes_mallows_score, rand_score, adjusted_rand_score
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

	# fowlkes_mallows score
	print(f"fowlkes_mallows score : {fowlkes_mallows_score(y, labels)}")

	# silhouette score
	print(f"silhouette score : {silhouette_score(X, labels)}")

	# rand score
	print(f"rand score : {rand_score(y, labels)}")

	# adjusted rand score
	print(f"adjusted rand score : {adjusted_rand_score(y, labels)}")


if __name__ == "__main__" :
	X, y = data.load_data("./Data/train", "train")

	x_dem = visualize.dimension_reduction(X, 2, "TSNE")

	labels_dem, centers_dem = kmeans(x_dem, 6, 20)
	# cluster using the orginal data
	labels_org, centers_org = kmeans(X, 6, 20)
	# create two figures at the same time
	# visualize.plot([x_dem, x_dem, x_dem], [labels_dem, labels_org, y], dim_target=3, need_dimension_reduction=False)
	
	metric("kmeans orginal", X, y, labels_org, centers_org)
	metric("kmeans dimension reduction", x_dem, y, labels_dem, centers_dem)