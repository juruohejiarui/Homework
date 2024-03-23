import math
import sys
import time

def multi(a, b) :
    a, b = str(a), str(b)
    n, m = 0, 0
    lsp = []
    f, g = [], []
    pi : float = math.acos(-1)
        
    def fft(f, flag) :
        for i in range(0, n) : 
            if i < lsp[i] : f[i], f[lsp[i]] = f[lsp[i]], f[i]
        i = 2
        while i <= n:
            lsl = i >> 1
            w1 = complex(math.cos(2 * pi / i), math.sin(2 * pi / i))
            if flag == 0: w1 = complex(w1.real, -w1.imag)
            for st in range(0, n, i) :
                buf = complex(1, 0)
                for k in range(st, st + lsl) :
                    tmp = buf * f[k + lsl]
                    f[k + lsl] = f[k] - tmp
                    f[k] += tmp
                    buf *= w1 
            i <<= 1
    n, m = len(a) - 1, len(b) - 1
    m += n
    n = 1
    while n <= m : n <<= 1
    lsp = [0 for i in range(0, n)]
    f = [complex(0, 0) for i in range(0, n)]
    g = [complex(0, 0) for i in range(0, n)]
    for i in range(0, len(a)) :
        f[i] = complex(float(a[-i - 1]), 0)
    for i in range(0, len(b)) :
        g[i] = complex(float(b[-i - 1]), 0)
    for i in range(1, n):
        if (i & 1) == 0 :
            lsp[i] = lsp[i >> 1] >> 1
        else: lsp[i] = (lsp[i >> 1] >> 1) | (n >> 1)
    fft(f, 1); fft(g, 1)
    for i in range(0, n) : f[i] *= g[i]
    fft(f, 0)
    shift = 0
    ans = [int(f[i].real / n + 0.49) for i in range(0, n)]
    ans.append(0)
    for i in range(0, n + 1) :
        ans[i] += shift
        shift = ans[i] // 10; ans[i] %= 10
    st = n
    while st > 0 and ans[st] == 0 : st -= 1
    ansnum = ''
    for i in range(st, -1, -1) : ansnum += str(ans[i])
    print(ansnum)

if __name__ == "__main__":
    sys.set_int_max_str_digits(0)
    while True :
        try : 
            a, b = input().split(' ')
            a, b = int(a), int(b)
            if a > 10**30000 and b > 10**30000 :
                print(a * b)
            else : multi(a, b)
        except Exception:
            break
