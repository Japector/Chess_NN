import os
import sys
import chess
import chess.engine
import numpy as np
import multiprocessing
from tqdm import tqdm
import argparse
import datetime
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.utility import board_to_array
from utils.npz_merge import merge_npz_datasets


STOCKFISH_PATH = "/usr/games/stockfish"
OUTPUT_NAME    = "Auto_generated_check"

NUM_RANDOM      = 0
NUM_MATE        = 0
NUM_CHECK       = 100000
NUM_DRAW        = 0
GENERATOR       = "stockfish"
RANDOM_MOVES    = 50



# Available options:
#   Mode:
#     - 'random'   : generate positions randomly
#     - 'mate'     : generate positions with mate
#     - 'check'    : generate positions where the king is in check
#     - 'draw'     : generate drawn positions
#
#   Generator:
#     - 'random'    : use random move selection
#     - 'stockfish' : use Stockfish to generate moves
#
#   Random moves:
#     - 50      : number of random moves, default is 50













def generate_position(mode = "random", generator = "random", limit = chess.engine.Limit(time=0.01), num_moves = 50): 
    board = chess.Board()

    if generator == "stockfish":

        with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:

            if mode == "random":
                for _ in range(np.random.randint(5, num_moves)):
                    result = engine.play(board, limit)
                    if result.move:
                        board.push(result.move)
                    if board.is_game_over():
                        break

            elif mode == "mate":
                while True:
                    board = chess.Board()

                    while not board.is_game_over():
                        result = engine.play(board, limit)
                        if result.move:
                            board.push(result.move)
                        if board.is_game_over():
                            break

                    if board.is_checkmate():
                        break

            elif mode == "check":
                while True:
                    board = chess.Board()

                    while not (board.is_game_over() or board.is_check()):
                        legal_moves = list(board.legal_moves)
                        if not legal_moves:
                            break
                        board.push(np.random.choice(legal_moves))

                    if board.is_check():
                        break    
                        
            elif mode == "draw":
                while True:
                    board = chess.Board()

                    while not (board.is_game_over() or board.is_stalemate()):
                        legal_moves = list(board.legal_moves)
                        if not legal_moves:
                            break
                        board.push(np.random.choice(legal_moves))

                    if board.is_stalemate():
                        break   
    else:

        if mode == "random":
            for _ in range(np.random.randint(5, num_moves)):
                moves = list(board.legal_moves)
                if moves:
                    board.push(np.random.choice(moves))
                if board.is_game_over():
                    break

        elif mode == "mate":
            while True:
                board = chess.Board()

                while not board.is_game_over():
                    legal_moves = list(board.legal_moves)
                    if not legal_moves:
                        break
                    board.push(np.random.choice(legal_moves))

                if board.is_checkmate():
                    break

        elif mode == "check":
            while True:
                board = chess.Board()

                while not (board.is_game_over() or board.is_check()):
                    legal_moves = list(board.legal_moves)
                    if not legal_moves:
                        break
                    board.push(np.random.choice(legal_moves))

                if board.is_check():
                    break    
        elif mode == "draw":
            while True:
                board = chess.Board()

                while not (board.is_game_over() or board.is_stalemate()):
                    legal_moves = list(board.legal_moves)
                    if not legal_moves:
                        break
                    board.push(np.random.choice(legal_moves))

                if board.is_stalemate():
                    break   
    return board


def evaluation(board):
    
    fen = board.fen()
    
    if board.is_checkmate():
        eval_score = 10000 if board.turn == chess.BLACK else -10000
        return eval_score / 1000.0, fen, {}             
    elif board.is_stalemate():
        return 0.0, fen, {}  
    
    
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:

        if board.is_check():
            limit = chess.engine.Limit(time = 0.1)
        else:
            limit = chess.engine.Limit(time = 0.1)

        info = engine.analyse(board, limit)
        score = info["score"].white()    

        if score is not None:
            if score.is_mate():
                eval_score = 10000 if score.mate() > 0 else -10000
            else: 
                eval_score = score.score()
            return eval_score / 1000.0, fen, info 
        return None



def process_sample(args):
    index, mode, generator, limit, num_moves = args
    board = generate_position(mode, generator, limit, num_moves)
    if board is None:
        return None

    board_array = board_to_array(board)
    eval_result = evaluation(board)
    if eval_result is None:
        return None
    eval_score, board_fen, board_info = eval_result

    return board_array, eval_score, board_fen, board_info



def generate_training_samples_parallel(num_samples = 1000, 
                                       mode = "random", 
                                       generator = "random", 
                                       limit = chess.engine.Limit(time=0.01), 
                                       num_moves = 30, 
                                       file_name = "AUTO", 
                                       eval_desc="AUTO"):

    args_list = [(index, mode, generator, limit, num_moves) for index in range(num_samples)]



    """Generate training samples in parallel using multiprocessing."""
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        results = list(tqdm(pool.imap(process_sample, args_list), total=num_samples, desc="Generating Samples"))

    
    results = [r for r in results if r is not None]

    
    X_train, y_train, board_fen, stockfish_info = zip(*results)

    
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    board_fen = np.array(board_fen, dtype=object)
    stockfish_info = np.array(stockfish_info, dtype=object)

    
    metadata = {
        "dataset": file_name,
        "mode": mode,
        "generator": generator,
        "limit": limit,
        "evaluation_strategy": eval_desc,
        "num_samples": len(X_train),
        "date": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    
    np.savez_compressed(f"{file_name}.npz", 
                        positions=X_train, 
                        scores=y_train, 
                        fens=board_fen,
                        stockfish_info=stockfish_info,
                        metadata=metadata)

    print(f"\nDataset saved as: {file_name}.npz")
    print(f"Metadata included in the .npz file")






data_dir = "./resources/Data/"
NUM_WORKERS = multiprocessing.cpu_count()

generation_settings = [
    {"num_samples": NUM_RANDOM, "mode": "random", "generator": GENERATOR, "file_name": f"{data_dir}data_random"},
    {"num_samples": NUM_CHECK, "mode": "check",  "generator": GENERATOR, "file_name": f"{data_dir}data_check"},
    {"num_samples": NUM_DRAW, "mode": "draw",   "generator": GENERATOR, "file_name": f"{data_dir}data_draw"},
    {"num_samples": NUM_MATE, "mode": "mate",   "generator": GENERATOR, "file_name": f"{data_dir}data_mate"},
]

generation_settings = [cfg for cfg in generation_settings if cfg["num_samples"] > 0]
file_names = [cfg["file_name"] + ".npz" for cfg in generation_settings]


if __name__ == "__main__":
    
    for config in generation_settings:
        generate_training_samples_parallel(
            limit=chess.engine.Limit(time=0.01),
            num_moves=RANDOM_MOVES,
            eval_desc="AUTO",
            **config
        )


    merge_npz_datasets(file_names, 
                    merged_file_name = f"{data_dir}{OUTPUT_NAME}")
    

    for file in file_names:
        if os.path.exists(file):
            os.remove(file)
