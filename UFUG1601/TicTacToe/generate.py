import numpy as np

valid_states = []

def to_int(state : np.ndarray) -> int :
    res = 0
    for i in range(0, 3):
        for j in range(0, 3):
            res = res * 3 + state[i, j]
    return res