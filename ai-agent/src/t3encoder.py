

def encode(board, player_id):
    # board (27) + player (3) hot encoded vector
    size = 30

    enc_array = [0] * size
    for i in range(len(board)):
        j = i * 3
        enc_array[j + board[i]] = 1

    enc_array[player_id + 27] = 1
    return enc_array
