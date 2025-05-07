import scipy
import numpy as np
import matplotlib.pylab as plt
import matplotlib
import sklearn as sk
from mpl_toolkits.mplot3d import Axes3D

import data

def plot(x, y, dim_input, dim_target = 3) :
    tsne = sk.manifold.TSNE(n_components=dim_target, random_state=42)
    x_tsne = tsne.fit_transform(x)
    print(x_tsne[0 : 5])
    print(x_tsne.shape)

    fig = plt.figure(figsize=(8, 8))
    ax = Axes3D(fig)

    ax.scatter(x_tsne[:, 0], x_tsne[:, 1], x_tsne[:, 2], 
               c=matplotlib.colormaps.get_cmap("plasma")(y / 6.0), s=2)

    # plt.legend()
    plt.show()

if __name__ == "__main__" :
    X, y = data.load_data("./Data/train", "train")
    print(X.shape, y.shape)
    plot(X, y, 3)