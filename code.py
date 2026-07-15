"""
code.py — Inference script for multi-person UWB radar localization.

Usage (as required by the assignment):
    python submission/code.py \
        --input-path  <path/to/input.npy> \
        --output-path <path/to/output.jsonl>

Input : .npy array of shape (T, 6, 3, 120, 2)  — raw CIR I/Q, 6 radars, 3 antennas, 120 range bins
Output: .jsonl, one object per frame: {"frame": i, "localizations": [[x, y], ...]}

Allowed dependencies only: NumPy, SciPy, TensorFlow.
The trained model (model.tflite) must sit next to this file.
"""

import argparse
import json
import os

import numpy as np
from scipy.ndimage import maximum_filter

# TensorFlow is only needed for the TFLite interpreter.
import tensorflow as tf

# ----------------------------------------------------------------------------- 
# Configuration — MUST match training (training/train.py)
# -----------------------------------------------------------------------------
ROOM_X, ROOM_Y = 4.8, 7.2          # room size in metres
GRID_X, GRID_Y = 12, 18            # heatmap resolution (0.4 m cells)
CELL = 0.4                         # cell size in metres

N_RADARS, N_ANT, N_BINS = 6, 3, 120
IN_ROWS = N_RADARS * N_ANT         # 18 "rows" (radar x antenna)

DETECT_THRESHOLD = 0.65            # heatmap peak threshold (tuned on validation)
MAX_PEOPLE = 4
TEMPORAL_WIN = 5                   # moving-average window over heatmaps (frames)
CLUTTER_ALPHA = 0.02               # EMA coefficient for adaptive clutter estimate


# -----------------------------------------------------------------------------
# Preprocessing (identical maths to training)
# -----------------------------------------------------------------------------
def cir_magnitude(iq):
    """(T, 6, 3, 120, 2) I/Q -> (T, 18, 120) magnitude."""
    mag = np.sqrt(iq[..., 0] ** 2 + iq[..., 1] ** 2)           # (T, 6, 3, 120)
    return mag.reshape(mag.shape[0], IN_ROWS, N_BINS)           # (T, 18, 120)


def preprocess_sequence(iq):
    """
    Build the 2-channel network input for every frame:
      ch0: clutter-removed magnitude (static background subtracted, EMA estimate)
      ch1: frame-to-frame difference (motion channel)
    Both channels are normalised per-frame by a robust scale.
    Returns float32 array of shape (T, 18, 120, 2).
    """
    mag = cir_magnitude(iq.astype(np.float32))                  # (T, 18, 120)
    T = mag.shape[0]

    out = np.empty((T, IN_ROWS, N_BINS, 2), dtype=np.float32)

    # Initialise clutter with the median of the first second of data
    # (robust to a moving person) then adapt slowly with an EMA.
    init = min(25, T)
    clutter = np.median(mag[:init], axis=0)                     # (18, 120)

    prev = mag[0]
    for t in range(T):
        cur = mag[t]

        fg = cur - clutter                                      # clutter removed
        diff = cur - prev                                       # motion

        # Robust per-frame normalisation (99th percentile of |fg|)
        scale = np.percentile(np.abs(fg), 99) + 1e-6
        out[t, ..., 0] = np.clip(fg / scale, -4.0, 4.0)
        out[t, ..., 1] = np.clip(diff / scale, -4.0, 4.0)

        clutter = (1.0 - CLUTTER_ALPHA) * clutter + CLUTTER_ALPHA * cur
        prev = cur

    return out


# -----------------------------------------------------------------------------
# Post-processing: heatmap -> list of (x, y)
# -----------------------------------------------------------------------------
def heatmap_to_positions(hm, threshold=DETECT_THRESHOLD, max_people=MAX_PEOPLE):
    """
    hm: (GRID_Y, GRID_X) float heatmap in [0, 1].
    Non-maximum suppression via 3x3 maximum filter, then sub-cell refinement
    with a centre-of-mass over the 3x3 neighbourhood of each peak.
    Returns list of [x, y] in metres, sorted by confidence, at most max_people.
    """
    local_max = (hm == maximum_filter(hm, size=3)) & (hm >= threshold)
    ys, xs = np.nonzero(local_max)
    if len(ys) == 0:
        return []

    order = np.argsort(hm[ys, xs])[::-1]
    positions, taken = [], []
    for idx in order:
        gy, gx = int(ys[idx]), int(xs[idx])

        # Suppress peaks closer than 2 cells (0.8 m) to an accepted one
        if any((gy - ty) ** 2 + (gx - tx) ** 2 < 4 for ty, tx in taken):
            continue

        # Sub-cell refinement: centre of mass on the 3x3 patch
        y0, y1 = max(0, gy - 1), min(GRID_Y, gy + 2)
        x0, x1 = max(0, gx - 1), min(GRID_X, gx + 2)
        patch = hm[y0:y1, x0:x1]
        w = patch.sum()
        if w > 0:
            yy, xx = np.mgrid[y0:y1, x0:x1]
            cy = float((yy * patch).sum() / w)
            cx = float((xx * patch).sum() / w)
        else:
            cy, cx = float(gy), float(gx)

        x_m = np.clip((cx + 0.5) * CELL, 0.0, ROOM_X)
        y_m = np.clip((cy + 0.5) * CELL, 0.0, ROOM_Y)
        positions.append([round(float(x_m), 3), round(float(y_m), 3)])
        taken.append((gy, gx))

        if len(positions) >= max_people:
            break

    return positions


# -----------------------------------------------------------------------------
# TFLite runner (handles full-INT8 models)
# -----------------------------------------------------------------------------
class TFLiteModel:
    def __init__(self, model_path):
        self.interp = tf.lite.Interpreter(model_path=model_path)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]

    def __call__(self, x):
        """x: (1, 18, 120, 2) float32 -> (GRID_Y, GRID_X) float heatmap."""
        if self.inp["dtype"] == np.int8:
            s, z = self.inp["quantization"]
            x = np.clip(np.round(x / s + z), -128, 127).astype(np.int8)
        self.interp.set_tensor(self.inp["index"], x)
        self.interp.invoke()
        y = self.interp.get_tensor(self.out["index"])
        if self.out["dtype"] == np.int8:
            s, z = self.out["quantization"]
            y = (y.astype(np.float32) - z) * s
        return y.reshape(GRID_Y, GRID_X)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "model.tflite")
    model = TFLiteModel(model_path)

    iq = np.load(args.input_path)                       # (T, 6, 3, 120, 2)
    assert iq.ndim == 5 and iq.shape[1:] == (N_RADARS, N_ANT, N_BINS, 2), \
        f"Unexpected input shape {iq.shape}"

    x = preprocess_sequence(iq)                         # (T, 18, 120, 2)
    T = x.shape[0]

    # Per-frame inference
    heatmaps = np.empty((T, GRID_Y, GRID_X), dtype=np.float32)
    for t in range(T):
        heatmaps[t] = model(x[t:t + 1])

    # Temporal smoothing: causal moving average over the heatmaps.
    # People move slowly relative to 25 Hz, this suppresses flicker.
    smoothed = np.empty_like(heatmaps)
    csum = np.cumsum(heatmaps, axis=0)
    for t in range(T):
        t0 = max(0, t - TEMPORAL_WIN + 1)
        window_sum = csum[t] - (csum[t0 - 1] if t0 > 0 else 0)
        smoothed[t] = window_sum / (t - t0 + 1)

    with open(args.output_path, "w") as f:
        for t in range(T):
            locs = heatmap_to_positions(smoothed[t])
            f.write(json.dumps({"frame": t, "localizations": locs}) + "\n")

    print(f"Wrote {T} frames to {args.output_path}")


if __name__ == "__main__":
    main()
