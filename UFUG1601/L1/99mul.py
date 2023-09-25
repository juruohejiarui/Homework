import traingle
if __name__ == "__main__":
    for i in range(1, 10):
        for j in range(i, 10):
            print("{0}*{1} = {2} ".format(i, j, i * j), end='')
        print("")
