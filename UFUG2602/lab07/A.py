def calc(arr : list, l : int, r : int) -> int :
    if l == r : return 0
    mid = (l + r) >> 1
    p1, p2 = l, mid + 1
    ans = calc(arr, l, mid) + calc(arr, mid + 1, r)
    tmp = []
    while p1 <= mid and p2 <= r:
        if arr[p1] <= arr[p2]:
            tmp.append(arr[p1]); p1 += 1
        else:
            tmp.append(arr[p2])
            ans += mid - p1 + 1; p2 += 1
    while p1 <= mid : tmp.append(arr[p1]); p1 += 1
    while p2 <= r : tmp.append(arr[p2]); p2 += 1
    for i in range(l, r + 1) : arr[i] = tmp[i - l]
    return ans

if __name__ == "__main__":
    n = int(input())
    arr = input().split(' ')
    for i in range(0, n): arr[i] = int(arr[i])
    print(calc(arr, 0, n - 1))