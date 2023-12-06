import numpy as np
import random
import math

def checkEnd(state : np.ndarray) -> int:
    for i in range(0, 3):
        if state[i, 0] == state[i, 1] and state[i, 1] == state[i, 2] and state[i, 0] != 0:
            return state[i, 0]
        elif state[0, i] == state[1, i] and state[1, i] == state[2, i] and state[0, i] != 0:
            return state[0, i]
    if state[0, 0] == state[1, 1] and state[1, 1] == state[2, 2] and state[2, 2] != 0:
        return state[0, 0]
    elif state[0, 2] == state[1, 1] and state[1, 1] == state[2, 0] and state[2, 0] != 0:
        return state[2, 0]
    return 0

def checkFull(state : np.ndarray) -> bool:
    for i in range(0, 3):
        for j in range(0, 3):
            if state[i, j] == 0: return False
    return True

def minimax(state : np.ndarray, isMax : bool) -> int:
    end = checkEnd(state)
    if end != 0: return end
    if checkFull(state): return 0

    if isMax:
        mxv = -math.inf
        for i in range(0, 3):
            for j in range(0, 3):
                if state[i, j] == 0:
                    state[i, j] = 1
                    mxv = max(mxv, minimax(state, not isMax))
                    state[i, j] = 0
        return mxv
    else:
        mnv = math.inf
        for i in range(0, 3):
            for j in range(0, 3):
                if state[i, j] == 0:
                    state[i, j] = -1
                    mnv = min(mnv, minimax(state, not isMax))
                    state[i, j] = 0
        return mnv

def bestMove(state : np.ndarray):
    ans, mxv = (0, 0), -math.inf
    for i in range(0, 3):
        for j in range(0, 3):
            if state[i, j] == 0:
                state[i, j] = 1
                tmp = minimax(state, False)
                state[i, j] = 0
                if tmp > mxv:
                    ans, mxv = (i, j), tmp
                elif tmp == mxv and random.randint(1, 2) == 1:
                    ans, mxv = (i, j), tmp
                

    return ans

def checkDangerous(board) -> (int, int):
    for i in range(0, 3):
        c = [0, 0, 0]
        c[board[i][0]] += 1
        c[board[i][1]] += 1
        c[board[i][2]] += 1
        if c[0] == 1 and (c[1] == 2 or c[2] == 2):
            for j in range(0, 3):
                if board[i][j] == 0: return (i, j)
    
    for i in range(0, 3):
        c = [0, 0, 0]
        c[board[0][i]] += 1
        c[board[1][i]] += 1
        c[board[2][i]] += 1
        if c[0] == 1 and (c[1] == 2 or c[2] == 2):
            for j in range(0, 3):
                if board[j][i] == 0: return (j, i)

    c = [0, 0, 0]
    for i in range(0, 3):
        c[board[i][i]] += 1
    if c[0] == 1 and (c[1] == 2 or c[2] == 2):
        for i in range(0, 3):
            if board[i][i] == 0: return (i, i)
    
    c = [0, 0, 0]
    for i in range(0, 3):
        c[board[i][2 - i]] += 1
    if c[0] == 1 and (c[1] == 2 or c[2] == 2):
        for i in range(0, 3):
            if board[i][2 - i] == 0: return (i, 2 - i)

    return (-1, -1)
def next_move(board):
    is_zero, is_full = True, True
    for i in range(0, 3):
        for j in range(0, 3):
            if board[i][j] != 0:
                is_zero = False
            else: is_full = False
    if is_zero : return (1, 1)
    elif is_full: return (0, 0)

    res = checkDangerous(board)
    if res != (-1, -1): return res

    state = np.zeros((3, 3))
    for i in range(0, 3):
        for j in range(0, 3):
            if board[i][j] == 1:
                state[i, j] = 1
            elif board[i][j] == 2:
                state[i, j] = -1
    ans = bestMove(state)
    return ans

# print(next_move([[0, 0, 0], [0, 2, 1], [2, 0, 0]]))