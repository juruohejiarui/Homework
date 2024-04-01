if __name__ == "__main__" :
    n, m = map(int, input().split())
    obj = []
    for i in range(n) :
        v, w = map(int, input().split())
        obj.append((v, w))
    f = [0 for i in range(0, m + 1)]
    for i in range(0, n) :
        for j in range(m, obj[i][1] - 1, -1) :
            f[j] = max(f[j], f[j - obj[i][1]] + obj[i][0])
    print(f[m])