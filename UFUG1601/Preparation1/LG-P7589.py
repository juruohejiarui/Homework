def solve():
    input()
    n, k, d = input().split(' ')
    n = int(n)
    x = 0
    for i in range(0, n):
        y, b, w = input().split(' ')
        b, w = int(b), int(w)
        x ^= abs(b - w) - 1
    if x == 0: print("No")
    else: print("Yes")
if __name__ == "__main__":
    T = int(input())
    while T > 0:
        solve()
        T -= 1