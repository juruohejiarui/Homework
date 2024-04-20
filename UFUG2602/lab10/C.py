if __name__ == "__main__" :
    n = int(input())
    edge = [[] for i in range(0, n + 1)]
    for i in range(0, n - 1) :
        u, v = map(int, input().split())
        edge[u].append(v)
        edge[v].append(u)
    
    w = [0] * (n + 1)
    
    q = int(input())
    for i in range(0, q) :
        x, r = map(int, input().split())
        w[x] += r
    
    def calcW(u : int, fa : int) :
        for v in edge[u] :
            if v == fa : continue
            w[v] += w[u]
            calcW(v, u)
    
    calcW(1, 0)
    for i in range(1, n + 1) :
        print(w[i], end = ' ')