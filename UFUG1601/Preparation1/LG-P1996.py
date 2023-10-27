if __name__ == "__main__":
    n, m = input().split(' ')
    n, m = int(n), int(m)
    vis = [0 for i in range(0, n)]
    p = -1
    ans = ""
    for i in range(0, n):
        for j in range(0, m):
            p = (p + 1) % n
            while vis[p] == 1: p = (p + 1) % n
        ans += f'{p + 1} '
        vis[p] = 1
    print(ans)