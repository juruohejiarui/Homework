import scipy
import numpy as np
import matplotlib.pylab as plt
import matplotlib
import sklearn as sk
from mpl_toolkits.mplot3d import Axes3D

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
        save_path = None,
        ) :
    
    if type(x) == np.ndarray : x = [x]
    if type(y) == np.ndarray : y = [y]
    if type(dim_target) == int : dim_target = [dim_target] * len(x)
    if type(need_dimension_reduction) == bool : need_dimension_reduction = [need_dimension_reduction] * len(x)

    if len(x) != len(y) or len(x) != len(dim_target) or len(x) != len(need_dimension_reduction) :
        raise ValueError("x, y, dim_target, need_dimension_reduction must have the same length")
    
    fig = plt.figure(figsize=(8, 8))
    # write each plot in a different subplot
    for i in range(len(x)) :
        if dim_target[i] == 2 :
            ax = fig.add_subplot(1, len(x), i + 1)
            if need_dimension_reduction[i] :
                x[i] = dimension_reduction(x[i], dim_target[i])
            ax.scatter(x[i][:, 0], x[i][:, 1], c=y[i], s=5)
            ax.set_title(f"Plot {i + 1}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
        elif dim_target[i] == 3 :
            ax = fig.add_subplot(1, len(x), i + 1, projection='3d')
            if need_dimension_reduction[i] :
                x[i] = dimension_reduction(x[i], dim_target[i])
            ax.scatter(x[i][:, 0], x[i][:, 1], x[i][:, 2], c=y[i], s=5)
            ax.set_title(f"Plot {i + 1}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

    plt.show()

    if save_path is not None :
        plt.savefig(save_path, dpi=300)

if __name__ == "__main__" :
    X, y = data.load_data("./Data/train", "train")

    plot(X, y, 3)