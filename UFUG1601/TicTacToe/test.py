import minimax_old as minimax
import zjj as rfs
import copy
import random

def result(board : list[list[int]]) -> int:
    for i in range(0, 3):
        if board[i][0] == board[i][1] and board[i][1] == board[i][2] and board[i][0] != 0:
            return board[i][0]
        if board[0][i] == board[1][i] and board[1][i] == board[2][i] and board[0][i] != 0:
            return board[0][i]
    if board[0][0] == board[1][1] and board[1][1] == board[2][2] and board[0][0] != 0:
        return board[0][0]
    if board[0][2] == board[1][1] and board[1][1] == board[2][0] and board[0][2] != 0:
        return board[0][2]
    return 0

def is_full(board : list[list[int]]) -> bool:
    for i in range(0, 3):
        for j in range(0, 3):
            if board[i][j] == 0: return False
    return True

if __name__ == "__main__":
    c = [0, 0, 0]
    for round in range(0, 50):
        board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        is_player1 = (random.randint(1, 2) == 1)
        if is_player1: print("is player 1   ")
        else: print("is player 2   ")
        turn = 0
        while result(board) == 0 and (not is_full(board)):
            turn += 1
            if turn % 2 == 1:
                if is_player1:
                    p = minimax.next_move(copy.deepcopy(board))
                else:
                    p = rfs.next_move(copy.deepcopy(board))
                board[p[0]][p[1]] = 1
            else:
                if not is_player1:
                    p = minimax.next_move(copy.deepcopy(board))
                else:
                    p = rfs.next_move(copy.deepcopy(board))
                board[p[0]][p[1]] = 2
            # print(board)
        res = result(board)
        if (not is_player1) and res != 0:
            res = 3 - res
        print(f"round {round + 1} : {res}")
        c[res] += 1
    
    print(c)

        