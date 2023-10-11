n = int(input())
for i in range(n):
    for j in range(2 * n - 1):
        # the range of *
        if j < n - i - 1 or j >= n + i:
            print('  ', end = '')
        else:
            print('* ', end = '')
    print('')