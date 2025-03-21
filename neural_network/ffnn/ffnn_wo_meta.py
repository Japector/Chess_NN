import os
import sys
import numpy as np
import tensorflow as tf
import tf2onnx
import datetime 
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras import Model, Input
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..")))
from utils.utility import extract_metadata_vector


MODEL_NAME = "ffnn_wo_meta"
TRAINING_DATA = "350k_gen_mix_60_20_10_10"



# -------------------------
# 0. Set  variables
# -------------------------

# Model files
KERAS   = f"./resources/models/{MODEL_NAME}.keras"
ONNX    = f"./resources/models/{MODEL_NAME}.onnx"

# Data and log directory
log_dir     = f"./resources/tensorboard/{MODEL_NAME}" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
data_dir    = "./resources/data/"

# -------------------------
# 1. Data preparation
# -------------------------

data = np.load(f"{data_dir}{TRAINING_DATA}.npz", allow_pickle=True)

X_train = data["positions"]
y_train = data["scores"] / 10
fens = data["fens"]
stockfish_info = data["stockfish_info"]
metadata = data["metadata"].item()

print("Shapes:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("fens:", fens.shape)
print("stockfish_info:", stockfish_info.shape)

print("\nMetadata:")
for key, value in metadata.items():
    print(f"{key}: {value}")

X_meta = np.array([extract_metadata_vector(fen) for fen in fens], dtype=np.float32)

# -------------------------
# 2. Model definition
# -------------------------

# Board input branch
board_input = Input(shape=(64,), name='board_input')
x = Dense(128, activation='relu')(board_input)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
x = BatchNormalization()(x)        
x = Dense(64, activation='relu')(x)
x = BatchNormalization()(x)       
x = Dropout(0.3)(x)
x = Dense(32, activation='relu')(x)
output = Dense(1)(x)

# Build model
model = Model(inputs=board_input, outputs=output)


# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

model.summary()

# ------------------------------------------
# 3. Callbacks (EarlyStopping + TensorBoard)
# ------------------------------------------

tensorboard_callback = TensorBoard(
    log_dir=log_dir, 
    histogram_freq=1, 
    write_graph=True, 
    write_images=True, 
    profile_batch=2
)


early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True) 


# ======= GRAPH TRACE LOGGINNG IN TENSORBOARD ========
@tf.function
def trace_model(x):
    return model(x)

# Trace on
tf.summary.trace_on(graph=True)

# Run
trace_model(X_train[:1])

with tf.summary.create_file_writer(log_dir).as_default():
    tf.summary.trace_export(
        name="model_trace_graph",
        step=0
    )

# -------------------------
# 4. Model Training
# -------------------------

history = model.fit(
    X_train, 
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=64,
    callbacks=[early_stop, tensorboard_callback],
    verbose=1
)

# -------------------------------
# 5. Saving model and create ONNX
# -------------------------------

model.save(KERAS)


spec = (
    tf.TensorSpec((None, 64), tf.float32, name="board_input"),   
)

model.output_names=['output']

# Convert Keras model to ONNX
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=ONNX)
print(f"Model converted to ONNX: {ONNX}")


# -------------------------
# 6. Evaluation
# -------------------------

loss, mae = model.evaluate(X_train, y_train)
print(f"\nTest MAE: {mae:.2f} cp (centipawn)")

# -------------------------
# 7. Predict example
# -------------------------

sample_pred = model.predict(X_train[:5])
print("\nPredicted centipawn values:", sample_pred.flatten() * 1000)

# -------------------------
# 8. Launch TensorBoard
# -------------------------

print("\nTo visualize training, run:\n")
print(f"    tensorboard --logdir {log_dir}\n")
print(f"    http://localhost:6006\n")
