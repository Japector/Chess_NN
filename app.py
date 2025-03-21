from flask import Flask, request, jsonify, render_template
import os
import chess
import chess.engine
from evaluation.play_game import get_move

app = Flask(__name__)


STOCKFISH_PATH = "/usr/games/stockfish"
board = chess.Board()

engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) 


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/models')
def list_models():
    model_folder = 'resources/models'
    model_files = [
        f for f in os.listdir(model_folder)
        if os.path.isfile(os.path.join(model_folder, f)) and f.endswith('.onnx')
    ]

    model_files.insert(0, "stockfish")
    model_files.insert(0, "user")

    return jsonify(model_files)


@app.route('/move', methods=['POST'])
def move():
    global board, engine
    data = request.get_json()

    model_white = data.get('model_white')
    model_black = data.get('model_black')

    try:
        if data['from'] + data['to'] != "AItrigger":

            user_move = chess.Move.from_uci(data['from'] + data['to'])
            if user_move not in board.legal_moves:
                return jsonify({'error': 'Illegal move', 'fen': board.fen()})

            board.push(user_move)



        if board.is_game_over():
            return jsonify({
                'fen': board.fen(),
                'status': 'Game over',
                'move': None
            })


        is_white_turn = board.turn == chess.WHITE
        ai_model = model_white if is_white_turn else model_black

        ai_move = get_move(board, ai_model, depth=3, is_white=is_white_turn, engine=engine)

        if ai_move:
            board.push(ai_move)


        info = engine.analyse(board, chess.engine.Limit(time=0.1))
        score = info["score"].white().score(mate_score=10000)

        return jsonify({
            'fen': board.fen(),
            'move': {
                'from': ai_move.uci()[:2],
                'to': ai_move.uci()[2:]
            },
            'eval': score
        })

    except Exception as e:
        return jsonify({'error': str(e), 'fen': board.fen()})


@app.route('/reset', methods=['POST'])
def reset():
    global board
    board.reset()
    return jsonify({'fen': board.fen()})


if __name__ == '__main__':
    app.run(debug=True)
