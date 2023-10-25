for a in range(111, 1000):
    b, c = 2 * a, 3 * a
    if b >= 1000 or c >= 1000:
        break
    s = str(a) + str(b) + str(c)
    vis = [0 for x in range(0, 10)]
    succ = True
    for p in range(0, 9):
        dig = int(s[p])
        vis[dig] += 1
        if dig == 0 or vis[dig] > 1:
            succ = False
            break
    if succ:
        print(f"{a} {b} {c}")