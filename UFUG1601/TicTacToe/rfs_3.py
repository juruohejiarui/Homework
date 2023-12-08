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

    def minimax(is_maximizing, depth, alpha, beta, b):
        if check_win(1, b):
            return 10 - depth
        if check_win(2, b):
            return depth - 10
        if all(b[i][j] != 0 for i in range(3) for j in range(3)):
            return 0

        if is_maximizing:
            best = -float('inf')
            for i in range(3):
                for j in range(3):
                    if b[i][j] == 0:
                        b[i][j] = 1
                        score = minimax(False, depth + 1, alpha, beta, b)
                        b[i][j] = 0
                        best = max(score, best)
                        alpha = max(alpha, best)
                        if beta <= alpha:
                            break
            return best
        else:
            best = float('inf')
            for i in range(3):
                for j in range(3):
                    if b[i][j] == 0:
                        b[i][j] = 2
                        score = minimax(True, depth + 1, alpha, beta, b)
                        b[i][j] = 0
                        best = min(score, best)
                        beta = min(beta, best)
                        if beta <= alpha:
                            break
            return best

    move = (-1, -1)
    score_best = -float('inf')
    alpha = -float('inf')
    beta = float('inf')
    for i in range(3):
        for j in range(3):
            if b[i][j] == 0:
                b[i][j] = 1
                score = minimax(False, 0, alpha, beta, b)
                b[i][j] = 0
                if score > score_best:
                    score_best = score
                    move = (i, j)
    return move

# print(next_move([[2,1,1],[1,1,2],[2,2,1]]))