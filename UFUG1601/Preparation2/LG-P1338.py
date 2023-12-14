import numpy as np

n, m = 0, 0
if __name__ == "__main__":
    n, m = input().split(' ')
    n, m, alsl = int(n), int(m), 0
    vis, als = np.zeros(n + 1, np.int32), np.zeros(n, np.int32)
    t = n * (n - 1) // 2
    for i in range(1, n + 1):
        t -= n - i
        # print(f"{t} {m}")
        if m > t :
            vis[i + m - t] = 1
            als[alsl] = i + m - t
            alsl += 1
            break
        als[alsl] = i
        alsl += 1
        vis[i] = 1
    for i in range(n, 0, -1):
        if vis[i] == 0:
            als[alsl] = i
            alsl += 1
    for i in range(0, n):
        print(als[i], end = ' ')
    print()