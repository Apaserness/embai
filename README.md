# EEAI Project — Multi-Person UWB Radar Localization (TinyML, ESP32-S3)

Estimates 2D positions of up to 4 people in a 4.8 m × 7.2 m room from raw UWB
radar CIR (6 radars × 3 antennas × 120 range bins, 25 Hz), deployable on an
ESP32-S3 via TensorFlow Lite for Microcontrollers.

## Repository layout

```
submission/
    code.py           # required inference script (numpy/scipy/tensorflow only)
    model.tflite      # produced by training/train.py (INT8)
training/
    train.py          # data loading, preprocessing, model, training, INT8 export
    tune_threshold.py # F1-based sweep of the detection threshold on held-out data
```

## How to reproduce

1. Download all 24 `.npz` acquisition windows from the HAEEAI Hugging Face
   dataset into `data/`.
2. Train and export the INT8 model:
   `python training/train.py --data-dir data/ --out submission/model.tflite`
3. Tune the detection threshold on the held-out windows and copy the best value
   into `DETECT_THRESHOLD` in `submission/code.py`:
   `python training/tune_threshold.py --data-dir data/ --model submission/model.tflite`
4. Run the official checks before every push:
   ```
   python submission/code.py --input-path evaluation/example/input_test.npy --output-path evaluation/example/my_output.jsonl
   python evaluation/evaluate_performance.py --gt-path evaluation/example/output_test.jsonl --pred-path evaluation/example/my_output.jsonl
   python evaluation/evaluate_constraint.py --model-path submission/model.tflite
   ```

## Approach

**Preprocessing.** The complex CIR is reduced to magnitude and reshaped to an
18 × 120 "image" (radar×antenna rows, range-bin columns). Two channels are fed
to the network: (1) magnitude with static clutter removed via a slowly-adapting
EMA background estimate, initialised with the median of the first second, and
(2) the frame-to-frame difference, which highlights moving people. Each frame
is normalised by its robust (99th-percentile) scale, making the model
insensitive to per-session gain differences — important because the final
evaluation uses an unseen acquisition session.

**Model.** A small CNN (4 conv blocks + 2 dense layers, ~340 k parameters)
regresses an 18 × 12 occupancy heatmap of the room (0.4 m cells) with a sigmoid
output. Targets are Gaussians (σ = 0.3 m) rendered at ground-truth positions;
the loss is a positively-weighted BCE to counter the background-heavy
heatmaps. Only TFLM-supported ops are used (Conv2D, BN folded at conversion,
Dense, ReLU) — no LSTM/GRU/RNN/custom/Flex ops.

**Post-processing (in `code.py`, off-model).** Heatmaps are causally averaged
over 5 frames (people move slowly relative to 25 Hz), then peaks are extracted
with 3 × 3 non-maximum suppression, a confidence threshold, a 0.8 m minimum
separation, and sub-cell refinement via centre-of-mass — yielding at most 4
`[x, y]` positions per frame, clipped to the room bounds required by the
format validator.

**Deployment constraints.** Full INT8 post-training quantization with a
representative dataset (int8 input and output). Expected model size ≈ 350 KB
(< 800 KB flash) and a tensor arena well under 300 KB — the largest activation
is the first conv output (18 × 60 × 16 int8 ≈ 17 KB).

**Validation.** Entire acquisition windows (the last 4) are held out, never
random frames, to mimic the unseen-session final evaluation and avoid
temporal leakage.

## Toggling notes

- Radar subset: fewer radars → smaller input, possibly better SNR per weight.
- Grid resolution (0.3 m cells) vs. arena/flash cost.
- Temporal window length and threshold (precision/recall trade-off at the
  1.0 m matching radius).
- Quantization-aware training if PTQ costs noticeable F1.
