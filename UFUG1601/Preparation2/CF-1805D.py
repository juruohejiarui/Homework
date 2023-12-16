import queue

n, E = 0, []
dis, dis2, vis = [], [], []
mxd, mxdp = 0, 0

def bfs(dis, st) :
    global mxd, mxdp
    dis[st] = 1
    vis = [0 for i in range(0, n + 1)]
    q = queue.Queue()
    q.put(st)
    vis[st] = 1
    while not q.empty() :
        u = q.get()
        if mxd < dis[u] :
            mxd, mxdp = dis[u], u
        for v in E[u] :
            if vis[v] : continue
            dis[v] = dis[u] + 1
            vis[v] = 1
            q.put(v)
    

if __name__ == "__main__":
    n = int(input())
    E = [[] for i in range(0, n + 1)]
    dis = [0 for i in range(0, n + 1)]
    dis2 = [0 for i in range(0, n + 1)]
    for i in range(1, n):
        u, v = map(int, input().split(' '))
        E[u].append(v)
        E[v].append(u)
    bfs(dis, 1)
    p1 = mxdp
    mxd = 0
    bfs(dis, p1)
    p2 = mxdp
    bfs(dis2, p2)

    cnt = [0 for i in range(0, n + 1)]
    ans = [0 for i in range(0, n + 1)]
    for i in range(1, n + 1):
        cnt[max(dis[i], dis2[i]) - 1] += 1
    ans[n] = n
    for i in range(n - 1, 0, -1):
        if cnt[i] == 0: ans[i] = ans[i + 1]
        elif ans[i + 1] == n : ans[i] = ans[i + 1] - cnt[i] + 1
        else : ans[i] = ans[i + 1] - cnt[i]
    for i in range(1, n + 1):
        print(f"{ans[i]} ", end = '')