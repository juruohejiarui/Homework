import numpy as np
import math
import random

state : np.ndarray
def get_init_dep() -> int:
    dep = 0
    for i in range(0, 3):
        for j in range(0, 3):
            if state[i, j] != 0: dep += 1
    return dep

def get_empty() -> [int]:
    pos = []
    for i in range(0, 3):
        for j in range(0, 3):
            if state[i, j] == 0:
                pos.append((i, j))
    return pos

def get_winner() -> int:
    for i in range(0, 3):
        if state[i, 0] == state[i, 1] and state[i, 1] == state[i, 2] and state[i, 0] != 0:
            return state[i, 0]
        if state[0, i] == state[1, i] and state[1, i] == state[2, i] and state[2, i] != 0:
            return state[0, i]
    if state[0, 0] == state[1, 1] and state[1, 1] == state[2, 2] and state[0, 0] != 0:
        return state[0, 0]
    if state[0, 2] == state[1, 1] and state[1, 1] == state[2, 0] and state[0, 2] != 0:
        return state[0, 2]
    return 0

def eval(dep : int) -> int:
    winner = get_winner()
    if winner == 0: return 0
    elif winner == 1: return 10 - dep
    else: return -10 + dep

def is_end() -> bool:
    return get_winner() != 0 or len(get_empty()) == 0

def minimax(dep : int, is_player1 : bool, alpha = -math.inf, beta = math.inf) -> (int, (int, int)):
    empty_pos = get_empty()
    if len(empty_pos) == 9: return (0, (1, 1))
    if is_end(): return (eval(dep), (0, 0))
    
    if is_player1:
        mxvl = -math.inf
        step = (-1, -1)
        for pos in empty_pos:
            state[pos[0], pos[1]] = 1
            vl = minimax(dep + 1, False, alpha, beta)[0]
            state[pos[0], pos[1]] = 0
            if vl > mxvl:
                mxvl, step = vl, pos
            alpha = max(alpha, vl)
            if alpha > beta: break
        return mxvl, step
    else:
        mnvl = math.inf
        step = (-1, -1)
        for pos in empty_pos:
            state[pos[0], pos[1]] = 2
            vl = minimax(dep + 1, True, alpha, beta)[0]
            state[pos[0], pos[1]] = 0
            if vl < mnvl:
                mnvl, step = vl, pos
            beta = min(beta, vl)
            if alpha > beta: break
        return mnvl, step

def best_move(is_player1):
    best_move = minimax(get_init_dep(), is_player1)[1]
    return best_move

def dangerous(board) -> (int, int):
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
    res = dangerous(board)
    if res != (-1, -1): return res
    global state
    state = np.zeros((3, 3))
    is_player1 = True
    c1 = 0
    for i in range(0, 3):
        for j in range(0, 3):
            if board[i][j] == 1:
                state[i, j] = 1
                c1 += 1
            elif board[i][j] == 2:
                state[i, j] = 2
                c1 += 1
    # for i in range(0, 4, 2):
    #     for j in range(0, 4, 2):
    #         if state[i, j] != 0 and state[2 - i, 2 - j] == 0:
    #             return (2 - i, 2 - j)
    # for i in range(0, 4, 2):
    #     for j in range(0, 4, 2):
    #         if state[i, j] == 0:
    #             return (i, j)
    if c1 % 2 == 1:
        is_player1 = False
        # for i in range(0, 3):
        #     for j in range(0, 3):
        #         if state[i, j] != 0: state[i, j] = 3 - state[i, j]
    
    if is_end(): return (0, 0)
    # print(state)
    ans = best_move(is_player1)
    return ans

# print(next_move([[0, 0, 0], [0, 0, 0], [1, 0, 0]]))

board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
x, y = next_move(board) 
assert x >= 0 and x <= 2
assert y >= 0 and y <= 2

board = [[0, 1, 0], [0, 0, 0], [0, 0, 0]]
x, y = next_move(board) 
assert x >= 0 and x <= 2
assert y >= 0 and y <= 2

board = [[0, 1, 2], [1, 2, 0], [1, 0, 2]]
x, y = next_move(board) 
assert x >= 0 and x <= 2
assert y >= 0 and y <= 2

board = [[0, 1, 2], [1, 2, 0], [1, 0, 0]]
x, y = next_move(board) 
assert x >= 0 and x <= 2
assert y >= 0 and y <= 2

board = [[1, 1, 2], [2, 2, 1], [1, 2, 0]]
x, y = next_move(board) 
assert x >= 0 and x <= 2
assert y >= 0 and y <= 2

board = [[1, 1, 2], [2, 2, 1], [1, 0, 0]]
x, y = next_move(board) 
assert x >= 0 and x <= 2
assert y >= 0 and y <= 2

board = [[1, 1, 2], [2, 2, 1], [1, 2, 1]]
x, y = next_move(board) 
assert x >= 0 and x <= 2
assert y >= 0 and y <= 2
print(next_move(board))
assert next_move(board) is not None
assert type(next_move(board)) != bool
assert isinstance(next_move(board), tuple)