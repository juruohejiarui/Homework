import numpy as np
matrix: np.ndarray

def sum(x1: int, y1: int, x2: int, y2: int):
    return matrix[x2, y2] - matrix[x2, y1 - 1] - matrix[x1 - 1, y2] + matrix[x1 - 1, y1 - 1]
if __name__ == "__main__":
    s = input().split(' ')
    n, m = int(s[0]), int(s[1])
    matrix = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        lst = input().split(' ')
        for j in range(1, m + 1):
            matrix[i, j] = np.int32(lst[j - 1])
            matrix[i, j] += matrix[i, j - 1] + matrix[i - 1, j] - matrix[i - 1, j - 1]
    
    for len in range(min(n, m), 0, -1):
        succ = False
        for x in range(1, n - len + 2):
            for y in range(1, m - len + 2):
                if sum(x, y, x + len - 1, y + len - 1) == len * len:
                    succ = True
                    break
            if succ :
                break
        if succ : 
            print(len)
            break