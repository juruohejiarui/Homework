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
                

    return ans

def next_move(board):
    is_zero = True
    for i in range(0, 3):
        for j in range(0, 3):
            if board[i][j] != 0:
                is_zero = False
    if is_zero : return (1, 1)
    state = np.zeros((3, 3))
    for i in range(0, 3):
        for j in range(0, 3):
            if board[i][j] == 1:
                state[i, j] = 1
            elif board[i][j] == 2:
                state[i, j] = -1
    ans = bestMove(state)
    return ans

# print(next_move([[1, 0, 0], [0, 2, 0], [0, 0, 0]]))