a, b, c, d = input().split('-')

if d == 'X':
    d = 10
else:
    d = int(d)

a += b + c

e = 0
for i in range(1, 10):
    e = (e + i * int(a[i - 1])) % 11

if e == d:
    print("Right")
else:
    if (e == 10) :
        print(f"{a[0 : 1]}-{a[1 : 4]}-{a[4 : 9]}-X")
    else:
        print(f"{a[0 : 1]}-{a[1 : 4]}-{a[4 : 9]}-{e}")