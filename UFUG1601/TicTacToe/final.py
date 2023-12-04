### Input an board (3x3 list) as function argument, 
# for example:
# board = [[0, 1, 0], [0, 0, 2], [0, 0, 0]], visualized as:
#  +---+---+---+
#  | 0 | 1 | 0 |
#  +---+---+---+
#  | 0 | 0 | 2 |
#  +---+---+---+
#  | 0 | 0 | 0 |
#  +---+---+---+
# 0: Empty
# 1: Piece of Play 1 (You)
# 2: Piece of Play 2 (Opponent)
# You need to complete the following function to return x, y, 
# which represents the position of next piece you place on the board.
# Don't place piece in the position that has been occupied, 
# otherwise, you will lose the game.

import random 
def next_move(board):
    ls = []
    for i in range(0, 3):
        for j in range(0, 3):
            if board[i][j] == 0:
                ls.append((i, j))
    x = random.randint(0, len(ls) - 1)
    return ls[x]