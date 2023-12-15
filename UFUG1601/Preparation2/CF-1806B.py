import math
def solve():
    n = int(input())
    a = list(map(int, input().split(' ')))
    cnt = [0, 0, 0]
    ans = 0
    for i in range(n):
        if a[i] > 1: cnt[2] += 1
        else: cnt[a[i]] += 1
    if cnt[0] <= math.ceil(n / 2):
        ans = 0
    elif cnt[0] == n or cnt[2] >= 1: ans = 1
    else: ans = 2
    print(ans)

if __name__ == "__main__":
    T = int(input())
    while T > 0:
        solve()
        T -= 1
