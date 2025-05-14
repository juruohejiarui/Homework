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

	labels_dem, centers_dem = kmeans(x_dem, 6, 20)
	# cluster using the orginal data
	labels_org, children_org = kmeans(X, 6, 20)
	# create two figures at the same time
	visualize.plot([x_dem, x_dem, x_dem], [labels_dem, labels_org, y], 2, need_dimension_reduction=False)
	