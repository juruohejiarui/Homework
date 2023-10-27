import math
if __name__ == "__main__":
    a, b = input().split(' ')
    a, b = int(a), int(b)
    ans = (math.floor(b / 4) - math.ceil(a / 4) + 1)
    if a % 2 == 0: a += 1
    if b % 2 == 0: b -= 1
    if a <= b: ans += (b - a) // 2 + 1
    print(int(ans))