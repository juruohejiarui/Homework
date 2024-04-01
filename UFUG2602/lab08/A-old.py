if __name__ == "__main__" :
    n = int(input())
    arr = list(map(int, input().split(' ')))
    f = [[-1 for i in range(0, n + 4)] for j in range(0, n + 4)]
    for i in range(1, n + 1) :
        f[i][i] = 0
    for len in range(2, n + 1) :
        for l in range(1, n - len + 2) :
            r = l + len - 1
            for k in range(l, r) :
                vl = f[l][k] + f[k + 1][r] + arr[l - 1] * arr[k] * arr[r]
                if f[l][r] == -1 or vl < f[l][r] : f[l][r] = vl
    print(f[1][n])