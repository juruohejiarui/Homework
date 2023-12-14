def solve():
    n = int(input())
    a = input().split(' ')
    for i in range(0, n):
        a[i] = int(a[i])
    same : bool = True
    for i in range(0, n - 1):
        if a[i] != a[i + 1]:
            same = False
            break
    if same : 
        print(1)
        return 
    als, ans = [0 for i in range(0, n)], 2
    als[0], als[1] = a[0], a[1]
    for i in range(2, n):
        if als[ans - 2] == als[ans - 1]: 
            als[ans - 1] = a[i]
            continue
        if als[ans - 1] == a[i]: continue
        if als[ans - 2] <= als[ans - 1]:
            if als[ans - 1] > a[i]:
                als[ans] = a[i]
                ans += 1
            else:
                als[ans - 1] = a[i]
        else:
            if als[ans - 1] < a[i]:
                als[ans] = a[i]
                ans += 1
            else:
                als[ans - 1] = a[i]
    print(ans)
if __name__ == "__main__":
    T = int(input())
    while T > 0:
        solve()
        T -= 1