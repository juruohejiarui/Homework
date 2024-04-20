if __name__ == "__main__" :
    n = int(input())
    preOrder = list(map(int, input().split()))
    inOrder = list(map(int, input().split()))

    p1 = [0] * (n + 1)
    p2 = [0] * (n + 1)
    for i in range(0, n) :
        p1[preOrder[i]], p2[inOrder[i]] = i, i

    ans = []
    
    def getAns(l1 : int, r1 : int, l2 : int, r2 : int) :
        if l1 == r1 : 
            ans.append(preOrder[l1])
            return 
        if l1 > r1 : return
        lSz = p2[preOrder[l1]] - l2
        getAns(l1 + 1, l1 + lSz, l2, l2 + lSz - 1)
        getAns(l1 + lSz + 1, r1, l2 + lSz + 1, r2)
        ans.append(preOrder[l1])
    
    getAns(0, n - 1, 0, n - 1)
    for i in range(0, n) :
        print(ans[i], end = ' ')
    print()