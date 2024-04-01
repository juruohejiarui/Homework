def multi(a, b) :
    l1, l2 = len(str(a)), len(str(b))
    if max(l1, l2) == 1: return a * b
    half = max(l1, l2) >> 1
    a0 = a % (10 ** half)
    a1 = a // (10 ** half)
    b0 = b % (10 ** half)
    b1 = b // (10 ** half)
    
    z0 = multi(a0, b0)
    z1 = multi(a1, b1)
    z2 = multi(a0 + a1, b0 + b1) - z0 - z1
    