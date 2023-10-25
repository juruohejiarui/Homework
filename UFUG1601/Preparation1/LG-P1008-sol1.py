for i in range(111111111, 1000000000):
    i_str = str(i)
    a, b, c = int(i_str[0 : 3]), int(i_str[3 : 6]), int(i_str[6 : 9])
    if a * 2 != b or a * 3 != c:
        continue
    vis = [0 for x in range(0, 10)]
    succ = True
    for p in range(0, 9):
        dig = int(i_str[p])
        vis[dig] += 1
        if vis[dig] > 1 or dig == 0:
            succ = False
            break
    if succ:
        print(f"{a} {b} {c}")
    