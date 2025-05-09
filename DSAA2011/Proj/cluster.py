from sklearn.cluster import KMeans
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

if __name__ == "__main__" :
	X, y = data.load_data("./Data/train", "train")

	x_dem = visualize.dimension_reduction(X, 2, "TSNE")

	labels_km, centers_km = kmeans(x_dem, 24, 20)
	labels_hc, centers_hc = hierarchical(x_dem, 24)
	# create two figures at the same time
	visualize.plot([x_dem, x_dem, x_dem], [labels_hc, labels_km, y], 2, need_dimension_reduction=False)
	