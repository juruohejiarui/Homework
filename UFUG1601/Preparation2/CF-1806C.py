import math

def solve():
    n = int(input())
    a = list(map(int, input().split(' ')))
    ans = 0
    for i in range(0, 2 * n):
        ans += abs(a[i])
    if n == 1 :
        ans = abs(a[1] - a[0])
    elif n == 2 :
        ans = min(abs(a[0] - 2) + abs(a[1] - 2) + abs(a[2] - 2) + abs(a[3] - 2), ans)
    
    if n % 2 == 0 :
        s = 0
        for i in range(0, 2 * n):
            s += abs(a[i] + 1)
        for i in range(0, 2 * n):
            ans = min(ans, s - abs(a[i] + 1) + abs(a[i] - n))
    print(ans)
if __name__ == "__main__":
    T = int(input())
    while T > 0:
        solve()
        T -= 1