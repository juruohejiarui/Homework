if __name__ == "__main__":
    n, W = map(int, input().split(' '))
    v, w, t = [0], [0], [0]
    for i in range(0, n):
        vi, wi, ti = map(int, input().split(' '))
        v.append(vi)
        w.append(wi)
        t.append(ti)
    
    f = [0 for i in range(0, W + 1)]
    for i in range(1, n + 1):
        if t[i] == 0:
            for j in range(W, w[i] - 1, -1) :
                f[j] = max(f[j], f[j - w[i]] + v[i])
        else:
            for j in range(w[i], W + 1) :
                f[j] = max(f[j], f[j - w[i]] + v[i])
    print(f[W])