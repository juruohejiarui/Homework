n = int(input())

for i in range(1, 2 * n):
    # use a formula to calculate the length
    for j in range(n - abs(n - i)):
        print('*', end = '')
    print('')
