if __name__ == "__main__" :
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    f = [0 for i in range(0, n + 1)]
    t = [0 for i in range(0, n + 1)]
    ls = []
    newB = []
    if n > m: n, m, a, b = m, n, b, a
    for i in range(0, m) :
        if b[i] > n : continue
        newB.append(b[i])
    b = newB
    for i in range(0, n) :
        f[a[i]] = i
    for i in range(0, n) :
        t[i] = f[b[i]]
    for i in range(0, n) :
        if len(ls) == 0 or t[i] > ls[-1] : ls.append(t[i])
        else :
            l, r = 0, len(ls) - 1
            while l < r :
                mid = (l + r) >> 1
                if ls[mid] > t[i] : r = mid
                else : l = mid + 1
            ls[l] = t[ i]
    print(len(ls))
