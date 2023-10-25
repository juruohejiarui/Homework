if __name__ == "__main__":
    n = int(input())
    a = input().split(' ')
    mp = {}
    vis = [0 for i in range(0, len(a))]
    for i in range(0, len(a)): 
        a[i] = int(a[i])
        mp[a[i]] = i
    for x in range(0, len(a)):
        for y in range(x + 1, len(a)):
            if a[x] + a[y] in mp:
                vis[mp[a[x] + a[y]]] = 1
    ans = 0
    for i in range(0, len(a)):
        if vis[i]:
            ans += 1
    print(ans)