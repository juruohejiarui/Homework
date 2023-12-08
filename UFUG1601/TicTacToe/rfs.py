def next_move(b):
    def check_win(player, b):
        # Check rows and columns for win
        for i in range(3):
            if all(b[i][j] == player for j in range(3)) or all(b[j][i] == player for j in range(3)):
                return True
        # Check diagonals for win
        if all(b[i][i] == player for i in range(3)) or all(b[i][2 - i] == player for i in range(3)):
            return True
        return False

    def minimax(is_maximizing, depth, b):
        # Check for win or draw
        if check_win(1, b):
            return 1
        if check_win(2, b):
            return -1
        if all(b[i][j] != 0 for i in range(3) for j in range(3)):
            return 0

        if is_maximizing:
            best = -float('inf')
            for i in range(3):
                for j in range(3):
                    if b[i][j] == 0:
                        b[i][j] = 1
                        score = minimax(False, depth + 1, b)
                        b[i][j] = 0
                        best = max(score, best)
            return best
        else:
            best = float('inf')
            for i in range(3):
                for j in range(3):
                    if b[i][j] == 0:
                        b[i][j] = 2
                        score = minimax(True, depth + 1, b)
                        b[i][j] = 0
                        best = min(score, best)
            return best

    # Finding the best move
    move = (-1, -1)
    score_best = -float('inf')
    for i in range(3):
        for j in range(3):
            if b[i][j] == 0:
                b[i][j] = 1
                score = minimax(False, 0, b)
                b[i][j] = 0
                if score > score_best:
                    score_best = score
                    move = (i, j)
    return move
