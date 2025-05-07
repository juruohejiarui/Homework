import numpy as np

def load_data(root_path, suffix) -> tuple[np.ndarray, np.ndarray]:
    X = np.loadtxt(f"{root_path}/X_{suffix}.txt")
    y = np.loadtxt(f"{root_path}/y_{suffix}.txt")
    return X, y

if __name__ == "__main__" :
    X, y = load_data("./Data/train", "train")
    print(X.shape)
    print(y.shape)