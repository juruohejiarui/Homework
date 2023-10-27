if __name__ == "__main__":
    n = int(input())
    a = input().split(' ')
    for i in range(0, n):
        a[i] = (int(a[i]), i)
    while a[-1] == '': a.pop(-1)
    a.sort()
    sum: float = 0
    ti: float = 0
    astr = ''
    for i in range(0, n):
        astr += f'{a[i][1] + 1} '
        sum += ti
        ti += a[i][0]
    print(astr)
    print("%.2f" % (sum / n))