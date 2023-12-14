import numpy as np
edge : list[list[int]]
vis : np.ndarray
col : np.ndarray
cnt : list[int] = [0, 0]

def dfs(u : int, fa : int) -> bool :
    vis[u], col[u] = 1, 1 ^ col[fa]
    cnt[col[u]] += 1
    for v in edge[u]:
        if vis[v] == 1:
            if col[u] ^ 1 != col[v]:
                return False
            continue
        res = dfs(v, u)
        if not res : return False
    return True

if __name__ == "__main__":
    n, m = input().split(' ')
    n, m = int(n), int(m)

    vis = np.zeros(n + 1, np.int32)
    col = np.zeros(n + 1, np.int32)
    edge = [[] for i in range(0, n + 1)]

    for i in range(0, m):
        u, v = input().split(' ')
        u, v = int(u), int(v)
        edge[u].append(v)
        edge[v].append(u)

    succ = True
    ans = 0
    for i in range(1, n + 1):
        if vis[i] == 0:
            cnt[0], cnt[1] = 0, 0
            res = dfs(i, 0)
            if not res : 
                succ = False
                break
            ans += min(cnt)
    if not succ:
        print('Impossible')
        exit()
    print(ans)
    