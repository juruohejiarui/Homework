import math
 
if __name__ == "__main__":
    n, m = map(int, input().split(' '))
    a = list(map(int, input().split(' ')))
    a = [0] + a
    qls = list(map(int, input().split(' ')))
    als = [0 for i in range(0, m)]
 
    b = [0 for i in range(n + 1)]
    mnb = [0 for i in range(n + 1)]
 
    a.sort()
    for i in range(1, n + 1):
        b[i] = a[i] - i + 1
        if i > 1: mnb[i] = min(mnb[i - 1], b[i])
        else: mnb[i] = b[i]
    s, s2 = 0, 0
    for i in range(1, n + 1):
        s += b[i] - mnb[n]
        s2 += b[i]
    # print(a)
    # print(mnb)
    for i in range(0, m):
        if n == 1:
            if qls[i] % 2 == 0:
                als[i] = a[1] - qls[i] // 2
            else:
                als[i] = a[1] - (qls[i] - 1) // 2 + qls[i]
            continue
        if qls[i] < n :
            als[i] = min(mnb[qls[i]] + qls[i], a[qls[i] + 1])
        else :
            if (qls[i] - n) % 2 == 0 :
                als[i] = mnb[n] - math.ceil(max(0, (qls[i] - n) // 2 - s) / n) + qls[i]
            else:
                mn = min(mnb[n - 1], a[n] - qls[i])
                s3 = s2 + n - 1 - qls[i] - mn * n
                als[i] = mn - math.ceil(max(0, (qls[i] - (n - 1)) // 2 - s3) / n) + qls[i]
    for i in range(0, m):
        print(f"{als[i]} ", end = '')