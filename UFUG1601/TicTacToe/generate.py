import minimax
import numpy as np

def to_state(x : np.ndarray) -> np.ndarray :
    res = np.zeros(x.shape)
    c1, c2 = 0, 0
    for i in range(0, 3):
        for j in range(0, 3):
            if x[i, j] == 1: c1 += 1
            elif x[i, j] == 2: c2 += 1
    if c1 > c2:
        for i in range(0, 3):
            for j in range(0, 3):
                if x[i, j] == 1: res[i, j] = -1
                elif x[i, j] == 2: res[i, j] = 1
    else:
        for i in range(0, 3):
            for j in range(0, 3):
                if x[i, j] == 1: res[i, j] = 1
                elif x[i, j] == 2: res[i, j] = -1
    return res

def to_int(x : np.ndarray) -> int :
    res = 0
    for i in range(0, 3):
        for j in range(0, 3):
            res = res * 3 + x[i, j]
    return int(res)

vis : set = set([])
dic = {}
def dfs_state(state : np.ndarray, t = 1) :
    id = to_int(state)
    if id in dic:
        return
    if minimax.checkFull(state):
        return
    dic[id] = minimax.bestMove(to_state(state))
    if t == 1:
        for i in range(0, 3):
            for j in range(0, 3):
                if state[i, j] == 0:
                    state[i, j] = 1
                    dfs_state(state, 2)
                    state[i, j] = 0
    else:
        for i in range(0, 3):
            for j in range(0, 3):
                if state[i, j] == 0:
                    state[i, j] = 2
                    dfs_state(state, 1)
                    state[i, j] = 0
    

if __name__ == "__main__":
    f = open("final.py", "w")
    dfs_state(np.zeros((3, 3)), 1)
    for key, val in dic.items():
        f.write(f"if key == {key} : return {val}\n")
    f.close()