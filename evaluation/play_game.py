import os
import sys
import chess
import chess.engine
import numpy as np
import onnxruntime as ort
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.utility import  board_to_array, board_to_multichannel_array, extract_metadata_vector, generate_move_maps





STOCKFISH_PATH = "/usr/games/stockfish"

MODEL_WHITE = "cnn_meta.onnx"
MODEL_BLACK = "stockfish"



# -------------------------
# 0. Set  variables
# -------------------------

move_to_int, int_to_move = generate_move_maps()


def detect_model_structure(session):

    input_info = session.get_inputs()
    input_names = [inp.name for inp in input_info]
    input_shape = input_info[0].shape

    
    spatial = len(input_shape) == 4 and input_shape[1] == 8 and input_shape[2] == 8

    
    meta = len(input_info) > 1

    
    output_shape = session.get_outputs()[0].shape

    if len(output_shape) == 1:
        move_gen = False  
    elif len(output_shape) == 2:
        move_gen = output_shape[1] != 1  
    else:
        move_gen = False  

    return {
        'spatial': spatial,
        'meta': meta,
        'move_gen': move_gen,
        'input_names': input_names
    }




model_sessions = {}
model_structures = {}

def get_model_session(model_name):
    if model_name not in model_sessions:
        model_path = os.path.join("resources", "models", model_name)
        session = ort.InferenceSession(model_path)
        model_sessions[model_name] = session
        model_structures[model_name] = detect_model_structure(session)
    return model_sessions[model_name], model_structures[model_name]


def evaluate_board(board, model_name, is_white = True):

    session, structure = get_model_session(model_name)
    input_names = structure['input_names']
    spatial = structure['spatial']
    move_gen = structure['move_gen']
    meta = structure['meta']

    if spatial:
        input_array = board_to_multichannel_array(board).reshape(1, 8, 8, 12).astype(np.float32)
    else:
        input_array = board_to_array(board).reshape(1, 64).astype(np.float32)

    if meta:
        meta_vector = extract_metadata_vector(board.fen()).reshape(1, -1).astype(np.float32)
        input_feed = {
            input_names[0]: input_array,
            input_names[1]: meta_vector
        }
    else:
        input_feed = {input_names[0]: input_array}
            
    result = session.run(None, input_feed)
    if move_gen: 
        return np.squeeze(result[0])
    else:
        value = float(np.squeeze(result[0]))
        return value if is_white else - value




def alphabeta(board, model_name, depth, alpha, beta, maximizing_player,  is_white = True):
    if depth == 0 or board.is_game_over():
        move_gen = False
        return evaluate_board(board, model_name, is_white), None

    best_move = None
    legal_moves = list(board.legal_moves)

    if maximizing_player:
        max_eval = -float('inf')
        for move in legal_moves:
            board.push(move)
            eval, _ = alphabeta(board, model_name, depth - 1, alpha, beta, False,   is_white)
            board.pop()
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in legal_moves:
            board.push(move)
            eval, _ = alphabeta(board, model_name, depth - 1, alpha, beta, True,  is_white)
            board.pop()
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_move


def get_user_move(board):
    move = None
    legal_moves = list(board.legal_moves)
    while move not in legal_moves:
        user_input = input("Your move (e.g., e2e4): ").strip()
        try:
            move_candidate = chess.Move.from_uci(user_input)
            if move_candidate in legal_moves:
                move = move_candidate
            else:
                print("Illegal move. Try again.")
        except:
            print("Invalid input format. Try again.")
    return move



def get_move(board, side_model, depth, is_white=True, engine=None):
    if side_model == "user":
        move = get_user_move(board)
    elif side_model == "stockfish":
        result = engine.play(board, chess.engine.Limit(time=0.1))
        move = result.move
    else:  
        _, structure = get_model_session(side_model)
        if structure['move_gen']:
            legal_moves = list(board.legal_moves)
            q_values = evaluate_board(board, side_model, is_white=is_white)
            move = max(legal_moves, key=lambda m: q_values[move_to_int.get(m.uci(), 0)])
        else:
            spatial = structure['spatial']
            _, move = alphabeta(board, side_model, depth, -float('inf'), float('inf'), True, is_white=is_white)
    return move



def play_game(model_white, model_black, ai_depth=3, stockfish_path="/usr/games/stockfish"):
    board = chess.Board()
    ai_white_captures = 0
    ai_black_captures = 0

    stockfish_engine = None
    if model_white == "stockfish" or model_black == "stockfish":
        stockfish_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    while not board.is_game_over():
        print("\nCurrent Board:")
        print(board)

        if board.turn == chess.WHITE: 
            move = get_move(board, model_white, ai_depth, is_white=True, engine=stockfish_engine)

            if move:
                captured = board.piece_at(move.to_square)
                if captured:
                    ai_white_captures += 1
                board.push(move)
                print(f"\n🔷 (White) plays: {move}")
        else: 
            move = get_move(board, model_black, ai_depth, is_white=True, engine=stockfish_engine)

            if move:
                captured = board.piece_at(move.to_square)
                if captured:
                    ai_black_captures += 1
                board.push(move)
                print(f"\n🔸 (Black) plays: {move}")

    print("\nGame Over")
    print(board)
    print(f"\nCaptures - AI White: {ai_white_captures}, AI Black: {ai_black_captures}")
    result = board.result()

    if result == "1-0":
        print("Winner: AI (White)")
    elif result == "0-1":
        print("Winner: AI (Black)")
    else:
        print("The game ended in a Draw")


if __name__ == "__main__":
    play_game(model_white= MODEL_WHITE, model_black=MODEL_BLACK, ai_depth=3)
