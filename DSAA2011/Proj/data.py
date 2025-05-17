import numpy as np
from sklearn.preprocessing import LabelBinarizer

def load_data(root_path, suffix) -> tuple[np.ndarray, np.ndarray]:
    X = np.loadtxt(f"{root_path}/X_{suffix}.txt")
    y = np.loadtxt(f"{root_path}/y_{suffix}.txt")
    return X, y

def load_label_name(root_path) -> dict[int, str] :
    with open(f"{root_path}/activity_labels.txt", "r") as f:
        lines = f.readlines()
        label_name = {}
        for line in lines:
            line = line.strip().split()
            label_name[int(line[0])] = line[1]
    
    return label_name

def label_binarize(y) -> np.ndarray :
    global handler
    if "handler" not in globals() or handler is None :
        handler = LabelBinarizer()
        _, y_train = load_data("./Data/train", "train")
        handler.fit(y_train)
    
    return handler.transform(y)

if __name__ == "__main__" :
    X, y = load_data("./Data/train", "train")
    label_name = load_label_name("./Data")
    print(X.shape)
    print(y.shape)
    print(label_name)