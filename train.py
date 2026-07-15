"""
train.py — Training pipeline for multi-person UWB radar localization (EEAI project).

Pipeline:
  1. Load the .npz acquisition windows downloaded from
     https://huggingface.co/datasets/HAEEAI/multi-person-localization
  2. Preprocess CIR I/Q -> 2-channel (clutter-removed magnitude + motion) input
  3. Train a small CNN that regresses a (18 x 12) occupancy heatmap of the room
  4. Convert to fully-quantized INT8 TFLite and save as submission/model.tflite

Usage:
    python training/train.py --data-dir data/ --out ../submission/model.tflite

Hold-out strategy: entire acquisition windows are held out for validation
(never random frames), because the final evaluation uses an unseen session.
"""

import argparse
import glob
import os

import numpy as np
import tensorflow as tf
os.environ["TF_USE_LEGACY_KERAS"] = "1"
# ----------------------------------------------------------------------------- 
# Configuration — MUST match submission/code.py
# -----------------------------------------------------------------------------
ROOM_X, ROOM_Y = 4.8, 7.2
GRID_X, GRID_Y = 12, 18            # 0.4 m cells
CELL = 0.4

N_RADARS, N_ANT, N_BINS = 6, 3, 120
IN_ROWS = N_RADARS * N_ANT
CLUTTER_ALPHA = 0.02

GT_SIGMA = 0.75                    # Gaussian target sigma, in cells (0.3 m)
BATCH = 128
EPOCHS = 40
LR = 1e-3
VAL_WINDOWS = 4                    # acquisition windows held out for validation
FRAME_STRIDE = 2                   # subsample training frames (25 Hz is redundant)


# -----------------------------------------------------------------------------
# Preprocessing (identical maths to submission/code.py)
# -----------------------------------------------------------------------------
def preprocess_sequence(iq):
    mag = np.sqrt(iq[..., 0] ** 2 + iq[..., 1] ** 2).astype(np.float32)
    mag = mag.reshape(mag.shape[0], IN_ROWS, N_BINS)
    T = mag.shape[0]

    out = np.empty((T, IN_ROWS, N_BINS, 2), dtype=np.float32)
    init = min(25, T)
    clutter = np.median(mag[:init], axis=0)
    prev = mag[0]
    for t in range(T):
        cur = mag[t]
        fg = cur - clutter
        diff = cur - prev
        scale = np.percentile(np.abs(fg), 99) + 1e-6
        out[t, ..., 0] = np.clip(fg / scale, -4.0, 4.0)
        out[t, ..., 1] = np.clip(diff / scale, -4.0, 4.0)
        clutter = (1.0 - CLUTTER_ALPHA) * clutter + CLUTTER_ALPHA * cur
        prev = cur
    return out


def make_heatmap_targets(people_xy, people_mask):
    """
    people_xy: (T, 4, 2) metres, people_mask: (T, 4) bool
    Returns (T, GRID_Y, GRID_X) float32 Gaussian heatmaps (max value 1 per person).
    """
    T = people_xy.shape[0]
    yy, xx = np.mgrid[0:GRID_Y, 0:GRID_X].astype(np.float32)
    hm = np.zeros((T, GRID_Y, GRID_X), dtype=np.float32)
    for t in range(T):
        for p in range(people_xy.shape[1]):
            if not people_mask[t, p]:
                continue
            x, y = people_xy[t, p]
            cx = x / CELL - 0.5
            cy = y / CELL - 0.5
            g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * GT_SIGMA ** 2))
            hm[t] = np.maximum(hm[t], g)   # max-combine overlapping people
    return hm


def load_dataset(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")
    print(f"Found {len(files)} acquisition windows")

    val_files = files[-VAL_WINDOWS:]
    train_files = files[:-VAL_WINDOWS]

    def load_files(file_list, stride):
        xs, ys = [], []
        for f in file_list:
            d = np.load(f)
            x = preprocess_sequence(d["radar_cir_iq"])
            y = make_heatmap_targets(d["people_xy"], d["people_mask"])
            xs.append(x[::stride])
            ys.append(y[::stride])
            print(f"  loaded {os.path.basename(f)}: {x.shape[0]} frames")
        return np.concatenate(xs), np.concatenate(ys)

    print("Loading training windows...")
    x_tr, y_tr = load_files(train_files, FRAME_STRIDE)
    print("Loading validation windows...")
    x_va, y_va = load_files(val_files, FRAME_STRIDE)
    print(f"train: {x_tr.shape}, val: {x_va.shape}")
    return x_tr, y_tr, x_va, y_va


# -----------------------------------------------------------------------------
# Model — small heatmap CNN, TFLM-friendly ops only
# (Conv2D / DepthwiseConv2D / Dense / BN / pooling / ReLU — no RNN/custom ops)
# -----------------------------------------------------------------------------
def build_model():
    inp = tf.keras.Input(shape=(IN_ROWS, N_BINS, 2))                 # (18,120,2)

    def conv(x, ch, k, s):
        x = tf.keras.layers.Conv2D(ch, k, strides=s, padding="same",
                                   use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        return tf.keras.layers.ReLU()(x)

    x = conv(inp, 16, (3, 5), (1, 2))     # (18, 60, 16)
    x = conv(x, 24, (3, 5), (2, 2))       # (9, 30, 24)
    x = conv(x, 32, (3, 3), (1, 2))       # (9, 15, 32)
    x = conv(x, 48, (3, 3), (3, 1))       # (3, 15, 48)

    x = tf.keras.layers.Flatten()(x)                                  # 2160
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(GRID_Y * GRID_X, activation="sigmoid")(x)
    out = tf.keras.layers.Reshape((GRID_Y, GRID_X))(x)

    return tf.keras.Model(inp, out)


def weighted_bce(y_true, y_pred):
    """BCE with extra weight on person cells (heatmaps are mostly background)."""
    eps = 1e-6
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
    pos_w = 1.0 + 15.0 * y_true
    bce = -(y_true * tf.math.log(y_pred) +
            (1.0 - y_true) * tf.math.log(1.0 - y_pred))
    return tf.reduce_mean(pos_w * bce)


# -----------------------------------------------------------------------------
# INT8 TFLite conversion
# -----------------------------------------------------------------------------
def convert_int8(model, rep_data, out_path):
    def rep_gen():
        idx = np.random.choice(len(rep_data), size=min(300, len(rep_data)),
                               replace=False)
        for i in idx:
            yield [rep_data[i:i + 1].astype(np.float32)]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_gen
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    tfl = conv.convert()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(tfl)
    print(f"Saved {out_path} ({len(tfl) / 1024:.1f} KB) — limit is 800 KB")


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True,
                    help="directory containing the dataset .npz files")
    ap.add_argument("--out", default="submission/model.tflite")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    x_tr, y_tr, x_va, y_va = load_dataset(args.data_dir)

    model = build_model()
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss=weighted_bce)

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5,
                                             monitor="val_loss"),
        tf.keras.callbacks.EarlyStopping(patience=8, monitor="val_loss",
                                         restore_best_weights=True),
    ]
    model.fit(x_tr, y_tr, validation_data=(x_va, y_va),
              batch_size=BATCH, epochs=args.epochs,
              shuffle=True, callbacks=callbacks)

    model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "training", "model_fp32.keras")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f"Saved {model_save_path}")
    convert_int8(model, x_tr, args.out)


if __name__ == "__main__":
    main()
