n: int
a: list = None
    
if __name__ == "__main__":
    n = int(input())
    a = input().split(' ')
    for i in range(0, n):
        a[i] = int(a[i])
    ans = False
    for i in range(0, n):
        if a[i] == 0:
            ans = ans or ((i + 1) % 2 == 0)
            break
    for i in range(n - 1, -1, -1):
        if a[i] == 0:
            ans = ans or ((n - i) % 2 == 0)
            break
    if ans : print("YES")
    else : print("NO")
