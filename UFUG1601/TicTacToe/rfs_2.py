def next_move(board):
    # Function to check if a player has won
    def check_win(b, player):
        # Check rows and columns
        for i in range(3):
            if all(b[i][j] == player for j in range(3)):
                return True
            if all(b[j][i] == player for j in range(3)):
                return True

        # Check diagonals
        if all(b[i][i] == player for i in range(3)):
            return True
        if all(b[i][2 - i] == player for i in range(3)):
            return True
        return False

    # Count how many pieces a player has on the board
    def count_pieces(b, player):
        count = 0
        for i in range(3):
            for j in range(3):
                if b[i][j] == player:
                    count += 1
        return count

    # Determine the current player
    player_1_pieces = count_pieces(board, 1)
    player_2_pieces = count_pieces(board, 2)
    player = 1 if player_1_pieces <= player_2_pieces else 2

    # If it's the first player and the center is free, take it
    if player == 1 and board[1][1] == 0:
        return 1, 1

    # Try a move to see if it wins
    def try_move(b, i, j, player):
        b[i][j] = player
        won = check_win(b, player)
        b[i][j] = 0
        return won

    # Look for a winning move
    for i in range(3):
        for j in range(3):
            if board[i][j] == 0:
                if try_move(board, i, j, player):
                    return i, j

    # Block opponent's winning move
    opponent = 1 if player == 2 else 2
    for i in range(3):
        for j in range(3):
            if board[i][j] == 0:
                if try_move(board, i, j, opponent):
                    return i, j

    # Otherwise, just take the first free spot
    for i in range(3):
        for j in range(3):
            if board[i][j] == 0:
                return i, j
