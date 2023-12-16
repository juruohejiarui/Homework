import numpy as np
import copy
import math

mod, n = 0, 0
def ksm(a : np.ndarray, k : int) -> np.ndarray :
    t = np.diag([1, 1, 1])
    mmtx = np.ones(a.shape, np.int64) * mod
    while k > 0:
        if k % 2 == 1: 
            t = t @ a % mmtx
            t = t % mmtx
        a = a @ a % mmtx
        k //= 2
    return t
            

if __name__ == "__main__":
    n, mod = map(int, input().split(' '))
    pw10 = [1 for i in range(0, 21)]
    for i in range(1, 21): pw10[i] = pw10[i - 1] * 10
    l = len(str(n))
    ans = np.zeros(3, np.int64)
    mmtx = np.ones(3, np.int64) * mod
    ans[2] = 1
    k = np.zeros((3, 3), np.int64)
    k[1, 0], k[1, 1], k[2, 0], k[2, 1], k[2, 2] = 1, 1, 1, 1, 1
    for i in range(1, l + 1):
        k[0, 0] = pw10[i] % mod
        if n < pw10[i] :
            ans = ans @ ksm(np.copy(k), n - pw10[i - 1] + 1) % mmtx
            break
        ans = ans @ ksm(np.copy(k), pw10[i] - pw10[i - 1]) % mmtx
    print(ans[0])