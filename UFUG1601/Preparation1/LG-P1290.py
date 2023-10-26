def solve(n : int, m : int) -> bool:
    if m == 0 : return False
    if n // m == 1: return not solve(m, n % m)
    return True
if __name__ == "__main__":
    T = int(input())
    while T > 0:
        N, M = input().split(' ')
        N, M = int(N), int(M)
        res = solve(max(N, M), min(N, M))
        if res == True: print("Stan wins")
        else: print("Ollie wins")
        T -= 1