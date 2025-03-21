import numpy as np
import chess



piece_values = {
    None: 0,
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 4,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 10,
}



piece_channel_map = {
    1: 0,   # White Pawn
    3: 1,   # White Knight
    4: 2,   # White Bishop
    5: 3,   # White Rook
    9: 4,   # White Queen
    10: 5,  # White King
    -1: 6,  # Black Pawn
    -3: 7,  # Black Knight
    -4: 8,  # Black Bishop
    -5: 9,  # Black Rook
    -9: 10, # Black Queen
    -10: 11 # Black King
}

piece_channel_map_board = {
    'P': 0,  
    'N': 1,
    'B': 2,
    'R': 3,
    'Q': 4,
    'K': 5,
    'p': 6,  
    'n': 7,
    'b': 8,
    'r': 9,
    'q':10,
    'k':11
}

def vector_to_multichannel(X_train):
    result = []

    for board_array in X_train:
        multi_channel = np.zeros((8, 8, 12), dtype=np.float32)

        for i in range(64):
            piece_val = board_array[i]
            if piece_val == 0:
                continue

            if piece_val in piece_channel_map:
                row = i // 8
                col = i % 8
                ch = piece_channel_map[piece_val]
                multi_channel[row, col, ch] = 1.0
        result.append(multi_channel)

    return np.array(result, dtype=np.float32)


def board_to_multichannel_array(board):
    array = np.zeros((8, 8, 12), dtype=np.float32)
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            row = i // 8
            col = i % 8
            ch = piece_channel_map_board.get((piece.symbol()))
            if ch is not None:
                array[row, col, ch] = 1.0
    return array


def board_to_array(board):
    arr = np.zeros(64)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece_values[piece.piece_type]
            if piece.color == chess.BLACK:
                value = -value
            arr[square] = value
    return arr.astype(np.float32)



def extract_metadata_vector(fen):
    board = chess.Board(fen)
    return np.array([
        1.0 if board.turn == chess.WHITE else 0.0,
        material_balance(board),
        board.fullmove_number / 100.0
    ], dtype=np.float32)

def material_balance(board):
    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    balance = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece_values.get(piece.piece_type, 0)
            balance += value if piece.color == chess.WHITE else -value
    return balance / 39.0




def generate_move_maps():
    files = 'abcdefgh'
    ranks = '12345678'
    move_to_int = {}
    int_to_move = {}
    index = 0

    for start_file in files:
        for start_rank in ranks:
            for end_file in files:
                for end_rank in ranks:
                    move = start_file + start_rank + end_file + end_rank
                    move_to_int[move] = index
                    int_to_move[index] = move
                    index += 1
 
    for file in files:
        for rank_start, rank_end in [('7', '8'), ('2', '1')]:
            for promotion_piece in ['q', 'r', 'b', 'n']:
                move = f"{file}{rank_start}{file}{rank_end}{promotion_piece}"
                move_to_int[move] = index
                int_to_move[index] = move
                index += 1

    return move_to_int, int_to_move



def flip_board(board):
    return board.transform(chess.flip_vertical)


def get_best_uci_move(fen, stockfish_path="stockfish", depth=10):
    board = chess.Board(fen)
    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        result = engine.analyse(board, chess.engine.Limit(depth=depth))
        best_move = engine.play(board, chess.engine.Limit(depth=depth)).move
        return best_move.uci()

def get_best_uci_move_stockfish(stockfish_info):
    if "pv" in stockfish_info and len(stockfish_info["pv"]) > 0:
        return stockfish_info["pv"][0].uci()
    else:
        return None