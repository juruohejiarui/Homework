import math
n = int(input())
# range_l, range_r which represents the range of '*'
rgl = rgr = math.ceil(n / 2)
for i in range(1, n + 1):
    for j in range(1, n + 1):
        if rgl <= j and j <= rgr:
            print('* ', end = '')
        else:
            print('  ', end = '')
    print('')
    # update the range
    if i <= n / 2: 
        rgl -= 1
        rgr += 1
    else:
        rgl += 1
        rgr -= 1