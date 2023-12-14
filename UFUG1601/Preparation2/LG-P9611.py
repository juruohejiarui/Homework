def f(x : int):
    ans = 0
    i, j = 1, 1
    while i <= x:
        j = x // (x // i)
        ans += (x // i) * (j - i + 1)
        i = j + 1
    return ans

    return ans
if __name__ == "__main__":
    l, r = input().split(' ')
    l, r = int(l), int(r)

    print(f(r) - f(l - 1))