import math
if __name__ == "__main__" :
    lst = list(map(int, input().split()))
    n = len(lst)
    
    lvlSz, pos, ans = 1, 0, 0
    while pos < n :
        nxtLvlPos = pos + lvlSz
        if nxtLvlPos >= n : break
        for i in range(0, lvlSz) :
            if nxtLvlPos + (i << 1) < n:
                ans += lst[i + pos] ^ lst[(i << 1) + nxtLvlPos]
            if nxtLvlPos + (i << 1) + 1 < n :
                ans += lst[i + pos] ^ lst[(i << 1) + nxtLvlPos + 1]
        pos = nxtLvlPos
        lvlSz <<= 1
        # print(f"pos = {pos}, ans = {ans}")
    print(ans)