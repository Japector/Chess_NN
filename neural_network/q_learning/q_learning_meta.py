import os
import sys
import chess
import chess.engine
import numpy as np
import random
import tf2onnx
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten, Concatenate
from collections import deque
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..")))
from utils.utility import board_to_multichannel_array, generate_move_maps, extract_metadata_vector




# ==== CONFIGURATION ====
ALPHA = 0.5 # Learning rate
GAMMA = 0.95 # Discont factor
EPSILON = 0.2 # Epsilon greedy
REPLAY_BUFFER_SIZE = 50000
BATCH_SIZE = 64
TRAINING_EPISODES = 100
TRAIN_INTERVAL = 4
MODEL_NAME = "q_learning_meta"
STOCKFISH_PATH = "/usr/games/stockfish"


# ==== 0. Set variables ====

# Model files
KERAS   = f"./resources/models/{MODEL_NAME}.keras"
ONNX    = f"./resources/models/{MODEL_NAME}.onnx"


# ==== ENEMY_MOVES ====
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)


# ==== REPLAY BUFFER ====
replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
loss_history = []
avg_reward_history = []



# ==== MODEL DEFINITION ====
def create_q_model(num_actions):
    board_input = Input(shape=(8, 8, 12), name='board_input')

    x = Conv2D(64, (5, 5), padding='same', activation='relu')(board_input)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(4, 4))(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)


    # Metadata input
    meta_input = Input(shape=(3,), name='meta_input')
    m = Dense(16, activation='relu')(meta_input)
    m = BatchNormalization()(m)

    # Combine both branches
    combined = Concatenate()([x, m])
    z = Dense(128, activation='relu')(combined)
    z = Dropout(0.3)(z)
    z = Dense(64, activation='relu')(z)
    output = Dense(num_actions, name='output')(z)

    # Build model
    model = Model(inputs=[board_input, meta_input], outputs=output)

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    return model



# ==== MOVE SELECTION ====
def select_move_epsilon_greedy(board, model, move_to_int, epsilon=EPSILON):
    legal_moves = list(board.legal_moves)
    X_meta = extract_metadata_vector(board.fen()).reshape(1, 3).astype(np.float32)
    if random.random() < epsilon:
        return random.choice(legal_moves)
    q_values = model.predict({  'board_input': board_to_multichannel_array(board).reshape(1, 8, 8, 12).astype(np.float32),
                                'meta_input': X_meta}, verbose=0)[0]
    best_move = max(legal_moves, key=lambda m: q_values[move_to_int.get(m.uci(), 0)]) 
    return best_move


# ==== REWARD FUNCTION ====
def calculate_reward(board, move, engine_eval=None):
    temp_board = board.copy()
    temp_board.push(move)

    if temp_board.is_checkmate():
        return 10.0
    if temp_board.is_stalemate():
        return 0.0  
    if temp_board.is_insufficient_material():
        return -2.0

    reward = 0.0
    if board.is_capture(move):
        reward += 1.0
    if board.gives_check(move):
        reward += 0.5


    if engine_eval:
        turn = board.turn 

        eval_before = engine.analyse(board, chess.engine.Limit(depth=10))["score"]
        eval_after  = engine.analyse(temp_board, chess.engine.Limit(depth=10))["score"]

        if turn == chess.WHITE:
            score_before = eval_before.white().score(mate_score=10000) or 0
            score_after = eval_after.white().score(mate_score=10000) or 0
        else:
            score_before = eval_before.black().score(mate_score=10000) or 0
            score_after = eval_after.black().score(mate_score=10000) or 0

        reward += (score_after - score_before) / 100.0

    return reward


# ==== EXPERIENCE COLLECTION ====
def collect_experience(board, model, move_to_int, stockfish = False):
    state_array = [board_to_multichannel_array(board).astype(np.float32),  np.array(extract_metadata_vector(board.fen()), dtype=np.float32)]
    move = select_move_epsilon_greedy(board, model, move_to_int)
    action_idx = move_to_int.get(move.uci())
    reward = calculate_reward(board, move, True)
    board.push(move)


    if not board.is_game_over():

        if stockfish:
            result = engine.play(board, chess.engine.Limit(time=0.01))       
            opponent_move = result.move
        else:
            opponent_move = np.random.choice(list(board.legal_moves))
        board.push(opponent_move)

    next_state_array = [board_to_multichannel_array(board).astype(np.float32),  np.array(extract_metadata_vector(board.fen()), dtype=np.float32)]
    done = board.is_game_over()
    replay_buffer.append((state_array, action_idx, reward, next_state_array, done))
    return reward

# ==== TRAINING STEP ====
def train_from_replay(model):
    if len(replay_buffer) < BATCH_SIZE:
        return None
    minibatch = list(replay_buffer)[-TRAIN_INTERVAL:]
    minibatch += random.sample(list(replay_buffer)[:-TRAIN_INTERVAL], BATCH_SIZE-TRAIN_INTERVAL)
    states, actions, rewards, next_states, dones = zip(*[(s, a, r, ns, d) for (s, a, r, ns, d) in minibatch])

    board_states, meta_states = zip(*states)
    next_board_states, next_meta_states = zip(*next_states)

    board_states = np.array(board_states, dtype=np.float32)
    meta_states = np.array(meta_states, dtype=np.float32)

    next_board_states = np.array(next_board_states, dtype=np.float32)
    next_meta_states = np.array(next_meta_states, dtype=np.float32)

    q_vals = model.predict([board_states, meta_states] , verbose=0)
    q_next = model.predict([next_board_states, next_meta_states], verbose=0)
    targets = np.copy(q_vals)
    for i in range(BATCH_SIZE):
        max_future_q = 0 if dones[i] else np.max(q_next[i])
        old_q = q_vals[i][actions[i]]  # Current Q(s,a)
        updated_q  = (1 - ALPHA) * old_q + ALPHA * (rewards[i] + GAMMA * max_future_q)
        targets[i][actions[i]] = updated_q # Override the current value
    history = model.fit([board_states, meta_states], targets, epochs=1, verbose=0) # Re-estimate
    return history.history['loss'][0]



# ==== TRAINING LOOP ====
def train_rl_agent(model=None, move_to_int=None, stockfish = False):
    all_moves = 0

    if model is None:
        if os.path.exists(KERAS):
            print(f"Loading existing model from {KERAS}")
            model = load_model(KERAS)
        else:
            print("No saved model found — training new model from scratch.")
            model = create_q_model(num_actions=len(move_to_int))

    training_counter = 0
    for episode in range(TRAINING_EPISODES):
        board = chess.Board()
        episode_rewards = []
        episode_loss = []
        game_moves = 0

        if episode % 2 != 0: # Changing colors between games to learn both.
            move = np.random.choice(list(board.legal_moves))
            board.push(move) 
        while not board.is_game_over():
            reward = collect_experience(board, model, move_to_int, stockfish)
            loss = None
            training_counter+=1
            if training_counter>=TRAIN_INTERVAL:
                loss = train_from_replay(model)
                training_counter = 0

            if loss is not None:
                episode_loss.append(loss)
            episode_rewards.append(reward)
            all_moves = all_moves + 1
            game_moves = game_moves + 1
        print(board)
        avg_reward = np.mean(episode_rewards)
        avg_loss = np.mean(episode_loss)
        avg_reward_history.append(avg_reward)
        loss_history.append(avg_loss)
        if loss is not None:
            print(f"Episode {episode+1}: Avg Reward = {avg_reward:.3f}  Loss = {avg_loss:.4f}")
        else:
            print(f"Episode {episode+1}: Avg Reward = N/A  Loss = N/A")
    model.save(KERAS)
    print(f"\n Model saved to {KERAS}")
    
    spec = (
        tf.TensorSpec((None, 8, 8, 12), tf.float32, name="board_input"),
        tf.TensorSpec((None, 3), tf.float32, name="meta_input")
    )
    model.output_names=['output']

    # Convert Keras model to ONNX
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=ONNX)


    model.summary()


# ==== ENTRY POINT  ====
if __name__ == "__main__":
    move_to_int, int_to_move = generate_move_maps()
    train_rl_agent(model=None, move_to_int=move_to_int, stockfish=True)
