import math

gbrd : list[list[int]]

def result() -> int:
    for i in range(0, 3):
        if gbrd[i][0] == gbrd[i][1] and gbrd[i][1] == gbrd[i][2] and gbrd[i][0] != 0:
            return gbrd[i][0]
        if gbrd[0][i] == gbrd[1][i] and gbrd[1][i] == gbrd[2][i] and gbrd[0][i] != 0:
            return gbrd[0][i]
    if gbrd[0][0] == gbrd[1][1] and gbrd[1][1] == gbrd[2][2] and gbrd[0][0] != 0:
        return gbrd[0][0]
    if gbrd[0][2] == gbrd[1][1] and gbrd[1][1] == gbrd[2][0] and gbrd[0][2] != 0:
        return gbrd[0][2]
    return 0

def is_full() -> bool:
    for i in range(0, 3):
        for j in range(0, 3):
            if gbrd[i][j] == 0: return False
    return True

ansx, ansy = 0, 0
def minimax(is_player1 : bool, dep : int) -> int:
    global ansx, ansy
    score = result()
    if score != 0: return -10 + dep
    if is_full(): return 0
    mxv, px, py = -114514, 0, 0
    for i in range(0, 3):
        for j in range(0, 3):
            if gbrd[i][j] == 0:
                if is_player1: gbrd[i][j] = 1
                else: gbrd[i][j] = 2
                vl = -minimax((not is_player1), dep + 1)
                gbrd[i][j] = 0
                if mxv < -vl:
                    mxv, px, py = -vl, i, j
    if dep == 0: ansx, ansy = px, py
    return mxv

                
    
def next_move(board):
    global gbrd
    gbrd = board
    c1 = 0
    is_player1 = True
    for i in range(3):
        for j in range(0, 3):
            if gbrd[i][j] != 0: c1 += 1
    if c1 % 2 != 0:
        is_player1 = False
        for i in range(3):
            for j in range(3):
                if gbrd[i][j] != 0: gbrd[i][j] = 3 - gbrd[i][j]
    if result() != 0 or is_full(): return (0, 0)
    print(f"is_player1 : {is_player1}")
    minimax(is_player1, 0)
    return (ansx, ansy)

# print(next_move([[0, 0, 0], [0, 0, 0], [2, 0, 0]]))