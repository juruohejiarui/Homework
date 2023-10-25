if __name__ == "__main__":
    word = input().lower()
    sentence = input().lower().split(' ')
    fir, cnt, pos = -1, 0, 0
    for i in range(0, len(sentence)):
        if sentence[i] == word:
            cnt += 1
            if fir == -1: fir = pos
        pos += len(sentence[i]) + 1
    if fir == -1:
        print('-1')
    else: 
        print(f"{cnt} {fir}")