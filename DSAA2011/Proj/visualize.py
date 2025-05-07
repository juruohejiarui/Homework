import scipy
import numpy as np
import matplotlib.pylab as plt
import matplotlib
import sklearn as sk
from mpl_toolkits.mplot3d import Axes3D

import data

def plot(x, y, dim_target = 3) :
    tsne = sk.manifold.TSNE(n_components=dim_target)
    x_tsne : np.ndarray = tsne.fit_transform(x)
    print(x_tsne.shape)

    if dim_target == 2 :
        plt.figure()
        plt.title(f"t-SNE 2")
        plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=plt.cm.Set1(y / 6), s=2)

    elif dim_target == 3 :

        fig = plt.figure()
        ax = Axes3D(fig)

        ax.set_title("t-SNE 3")
        ax.scatter(x_tsne[:, 0], x_tsne[:, 1], x_tsne[:, 2], 
                c=plt.cm.Set1(y / 6))

    plt.show()

if __name__ == "__main__" :
    X, y = data.load_data("./Data/train", "train")
    print(X.shape, y.shape)


    plot(X, y, 3)