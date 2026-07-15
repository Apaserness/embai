"""
tune_threshold.py — Sweep the heatmap detection threshold on held-out
acquisition windows and report the global F1 (same matching logic as
evaluate_performance.py: Hungarian assignment, TP if distance <= 1.0 m).

Usage:
    python training/tune_threshold.py --data-dir data/ \
        --model ../submission/model.tflite
"""

import argparse
import glob
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

# Reuse the exact preprocessing / postprocessing from the submission script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "submission"))
import code as sub  # submission/code.py

MATCH_DIST = 1.0
VAL_WINDOWS = 4


def f1_for_threshold(all_hm, all_gt, all_mask, thr):
    tp = fp = fn = 0
    for hm, gt, mask in zip(all_hm, all_gt, all_mask):
        preds = sub.heatmap_to_positions(hm, threshold=thr)
        gts = gt[mask.astype(bool)]
        if len(preds) == 0:
            fn += len(gts)
            continue
        if len(gts) == 0:
            fp += len(preds)
            continue
        P = np.array(preds)
        D = np.linalg.norm(P[:, None, :] - gts[None, :, :], axis=-1)
        ri, ci = linear_sum_assignment(D)
        matched = D[ri, ci] <= MATCH_DIST
        m = matched.sum()
        tp += m
        fp += len(preds) - m
        fn += len(gts) - m
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return f1, prec, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model", default="submission/model.tflite")
    args = ap.parse_args()

    model = sub.TFLiteModel(args.model)
    files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))[-VAL_WINDOWS:]

    all_hm, all_gt, all_mask = [], [], []
    for f in files:
        d = np.load(f)
        x = sub.preprocess_sequence(d["radar_cir_iq"])
        hms = np.stack([model(x[t:t + 1]) for t in range(len(x))])
        # same causal temporal smoothing as code.py
        csum = np.cumsum(hms, axis=0)
        for t in range(len(hms)):
            t0 = max(0, t - sub.TEMPORAL_WIN + 1)
            w = csum[t] - (csum[t0 - 1] if t0 > 0 else 0)
            all_hm.append(w / (t - t0 + 1))
        all_gt.extend(d["people_xy"])
        all_mask.extend(d["people_mask"])
        print(f"processed {os.path.basename(f)}")

    print(f"\n{'thr':>5} {'F1':>7} {'prec':>7} {'rec':>7}")
    best = (0, None)
    for thr in np.arange(0.15, 0.65, 0.05):
        f1, p, r = f1_for_threshold(all_hm, all_gt, all_mask, thr)
        print(f"{thr:5.2f} {f1:7.4f} {p:7.4f} {r:7.4f}")
        if f1 > best[0]:
            best = (f1, thr)
    print(f"\nBest threshold: {best[1]:.2f} (F1={best[0]:.4f})")
    print("Update DETECT_THRESHOLD in submission/code.py accordingly.")


if __name__ == "__main__":
    main()
