if __name__ == "__main__":
    # solution 2
    for i in range(1, 6):
        print("*" * i)
    for i in range(-4, 0):
        print("*" * (-i))
    i = 1
    # solution 2
    while i < 10:
        t = i
        if i > 5: t = 10 - i
        j = 1
        while j <= t:
            print("*", end = '')
            j += 1
        i += 1
    # solution 3
    for i in range(1, 10):
        print("*" * (5 - abs(i - 5)))
    
    