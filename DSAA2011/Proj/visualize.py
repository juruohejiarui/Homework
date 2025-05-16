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
        name : list[str] | str | None = None,
        label : dict[str] | None = None,
        save_path = None,
        ) :
    
    if type(x) == np.ndarray : x = [x]
    if type(y) == np.ndarray : y = [y] * len(x)
    if type(dim_target) == int : dim_target = [dim_target] * len(x)
    if type(name) == str : name = [name] * len(x)

    if len(x) != len(y) or len(x) != len(dim_target) :
        raise ValueError("x, y, dim_target, need_dimension_reduction must have the same length")
    
    nrow = 1 if len(x) <= 3 else 2
    ncol = int(math.ceil(len(x) / nrow))

    fig = plt.figure(figsize=(10, 8))

    if name is None :
        name = [f"Plot {i}" for i in range(len(x))]

    # write each plot in a different subplot
    for i in range(len(x)) :
        
        if dim_target[i] == 2 :
            ax = fig.add_subplot(nrow, ncol, i + 1)
            if label is not None :
                y_min = np.min(y[i])
                y_max = np.max(y[i])

                cmap = matplotlib.colormaps.get_cmap('viridis')
                colors = [cmap((i - y_min) / (y_max - y_min)) for i in range(int(y_min), int(y_max) + 1)]
                for j in range(int(y_min), int(y_max) + 1) :
                    ax.scatter(x[i][y[i] == j, 0], x[i][y[i] == j, 1], s=2, color=colors[j - int(y_min)], label=label[j])
            else :
                ax.scatter(x[i][:, 0], x[i][:, 1], c=y[i], s=2)
        elif dim_target[i] == 3 :
            ax = fig.add_subplot(nrow, ncol, i + 1, projection='3d')
            if label is not None :
                y_min = np.min(y[i])
                y_max = np.max(y[i])

                cmap = matplotlib.colormaps.get_cmap('viridis')
                colors = [cmap((i - y_min) / (y_max - y_min)) for i in range(int(y_min), int(y_max) + 1)]
                for j in range(int(y_min), int(y_max) + 1) :
                    ax.scatter(x[i][y[i] == j, 0], x[i][y[i] == j, 1], x[i][y[i] == j, 2], s=2, color=colors[j - int(y_min)], label=label[j])
            else :
                ax.scatter(x[i][:, 0], x[i][:, 1], x[i][:, 2], c=y[i], s=2)
            
            ax.set_zlabel("Z")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(name[i])
        if label is not None :
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    plt.tight_layout()

    if save_path is not None :
        plt.savefig(save_path, dpi=300)
    else :
        plt.show()

if __name__ == "__main__" :
    X, y = data.load_data("./Data/train", "train")
    X_tsne = dimension_reduction(X, 2, "TSNE")
    X_pca = dimension_reduction(X, 2, "PCA")
    plot([X_tsne], [y], dim_target=2, name="tsne", label=data.load_label_name("./Data"), save_path="visualization.png")