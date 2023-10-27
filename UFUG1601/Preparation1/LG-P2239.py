def calc(x: int):
    return (x - 1) * 4

if __name__ == "__main__":
    s = input().split(' ')
    n, x, y = int(s[0]), int(s[1]), int(s[2])
    t = min(x, y, n - x + 1, n - y + 1)
    ans = 1
    # print(t)
    for i in range(1, t):
        ans += calc(n - (i - 1) * 2)
    mv = [[0,1], [1,0], [0,-1], [-1,0]]
    px, py, dir = t, t, 0
    while px != x or py != y:
        ans += 1
        nx = px + mv[dir][0]
        ny = py + mv[dir][1]
        if nx < t or ny < t or nx > n - t + 1 or ny > n - t + 1:
            dir += 1
            nx = px + mv[dir][0]
            ny = py + mv[dir][1]
            # print(f"{px}, {py} {ans}")
        px, py = nx, ny
    print(ans)
        
    