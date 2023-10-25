def calcdig(dig : int) -> int:
    if dig == 0 or dig == 6 or dig == 9:
        return 6
    elif dig == 1:
        return 2
    elif dig == 2 or dig == 3 or dig == 5:
        return 5
    elif dig == 4:
        return 4
    elif dig == 7:
        return 3
    elif dig == 8:
        return 7
def calc(s : str) -> int:
    res = 0
    for i in range(0, len(s)):
        res += calcdig(int(s[i]))
    return res

ans, lim = 0, 0
def dfs(t : int, s : int, A : int, B : int) :
    global ans, lim
    if B > 0:
        if calc(str(A + B)) + s == lim:
            ans += 1
    elif s * 2 + 6 == lim:
        ans += 1
    if (t == 1 and A == 0) or (t == 2 and B == 0):
        rg = range(1, 10)
    else : rg = range(0, 10)
    for i in rg:
        s_nxt = s + calcdig(i)
        if s_nxt >= lim:
            continue
        if t == 1:
            dfs(t, s_nxt, A * 10 + i, B)
            dfs(2, s_nxt, A * 10 + i, B)
        else:
            dfs(2, s_nxt, A, B * 10 + i)
        
if __name__ == "__main__":
    lim = int(input()) - 4
    dfs(1, 0, 0, 0)
    print(ans)
    