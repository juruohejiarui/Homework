import collections
if __name__ == "__main__" :
    n, l, r = map(int, input().split(' '))
    a = list(map(int, input().split(' ')))
    
    mna = min(a)

    f = [-1000 * n for i in range(0, n + 1)]
    f[0] = a[0]
    q = collections.deque()
    for i in range(1, n + 1):
        while len(q) > 0 and q[0] < i - r : q.popleft()
        if i - l >= 0:
            while len(q) > 0 and f[q[-1]] <= f[i - l] : q.pop()
            q.append(i - l)
        if len(q) > 0 : f[i] = a[i] + f[q[0]]
    ans = -1000 * n
    for i in range(max(0, n + 1 - r), n + 1) :
        ans = max(ans, f[i])
    print(ans)
