def solve():
    n = int(input())
    if n % 6 == 0 : print("Roy wins!")
    else : print("October wins!")
if __name__ == "__main__":
    T = int(input())
    while T > 0:
        solve()
        T -= 1