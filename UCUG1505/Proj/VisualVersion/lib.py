import numpy as np

def intersection(A : np.ndarray, B : np.ndarray, C : np.ndarray, D : np.ndarray) -> np.ndarray | None :
    d1, d2 = B - A, D - C

    A_mat = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]]).reshape(2, 2)
    B_vec = C - A
    # B_vector = np.array([x3 - x1, y3 - y1])

    if np.linalg.det(A_mat) == 0:
        return None  # 平行或重合，无交点或无唯一交点

    t_s = np.linalg.solve(A_mat, B_vec)
    t = t_s[0]

    return A + t * d1