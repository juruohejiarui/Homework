import scipy
import numpy as np
import matplotlib.pylab as plt
import matplotlib
import sklearn as sk
from mpl_toolkits.mplot3d import Axes3D
import math

import data

def dimension_reduction(x, dim_target = 3, algorithm = "PCA") :
    if algorithm == "PCA" :
        pca = sk.decomposition.PCA(n_components=dim_target)
        x = pca.fit_transform(x)
    elif algorithm == "TSNE" :
        tsne = sk.manifold.TSNE(n_components=dim_target)
        x = tsne.fit_transform(x)

    return x

def plot(
        x = list[np.ndarray] | np.ndarray, 
        y = list[np.ndarray] | np.ndarray, 
        dim_target : int | list[int] = 3, 
        need_dimension_reduction : bool | list[bool] = True,
        name : list[str] | None = None,
        save_path = None,
        ) :
    
    if type(x) == np.ndarray : x = [x]
    if type(y) == np.ndarray : y = [y]
    if type(dim_target) == int : dim_target = [dim_target] * len(x)
    if type(need_dimension_reduction) == bool : need_dimension_reduction = [need_dimension_reduction] * len(x)

    if len(x) != len(y) or len(x) != len(dim_target) or len(x) != len(need_dimension_reduction) :
        raise ValueError("x, y, dim_target, need_dimension_reduction must have the same length")
    
    nrow = 1 if len(x) <= 3 else 2
    ncol = int(math.ceil(len(x) / nrow))

    fig = plt.figure(figsize=(8, 8))

    if name is None :
        name = [f"Plot {i}" for i in range(len(x))]

    # write each plot in a different subplot
    for i in range(len(x)) :
        if dim_target[i] == 2 :
            ax = fig.add_subplot(nrow, ncol, i + 1)
            if need_dimension_reduction[i] :
                x[i] = dimension_reduction(x[i], dim_target[i])
            ax.scatter(x[i][:, 0], x[i][:, 1], c=y[i], s=5)
            ax.set_title(name[i])
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
        elif dim_target[i] == 3 :
            ax = fig.add_subplot(nrow, ncol, i + 1, projection='3d')
            if need_dimension_reduction[i] :
                x[i] = dimension_reduction(x[i], dim_target[i])
            ax.scatter(x[i][:, 0], x[i][:, 1], x[i][:, 2], c=y[i], s=5)
            ax.set_title(name[i])
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

    if save_path is not None :
        plt.savefig(save_path, dpi=300)
    else :
        plt.show()

if __name__ == "__main__" :
    X, y = data.load_data("./Data/train", "train")

    plot(X, y, 3)