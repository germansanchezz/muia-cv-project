# ffNN Experiment Progress

## Objective

The current work covers the first image-recognition phase: designing and
training a feedforward neural network for the 13-class `xview_recognition`
dataset. CNNs, transfer learning and regularization techniques have not been
used.

The main comparison metrics are the validation metrics also used for tracking
performance against Codabench:

- Mean Accuracy
- Mean Recall
- Mean Precision

The validation split and data protocol have been kept fixed across the
experiments.

## Work completed today

### Experiment 01: baseline

Architecture:

```text
Flatten -> ReLU -> Dense(13) -> Softmax
```

Configuration:

- Adam, learning rate `1e-3`.
- Batch size `16`.
- Maximum `20` epochs.
- Original callbacks monitoring `val_accuracy`.
- Images without additional normalization.

Results:

| Metric | Result |
|---|---:|
| Mean Accuracy | 38.560% |
| Mean Recall | 30.950% |
| Mean Precision | 31.433% |

The best validation model was obtained at epoch 11.

### Experiment 02: one hidden dense layer

Change from the baseline:

```text
Flatten -> Dense(64, ReLU) -> Dense(13) -> Softmax
```

All other parameters were kept equal to the baseline. The model collapsed to
the majority class, `Building`, and completed all 20 epochs without improving.

| Metric | Result |
|---|---:|
| Mean Accuracy | 19.360% |
| Mean Recall | 7.692% |
| Mean Precision | 1.489% |

The confusion matrix showed 100% recall for `Building` and zero recall for all
other classes. This showed that simply adding a hidden layer was not a useful
direction under the original input scale and learning rate.

### Experiment 03: normalized inputs and lower learning rate

The `Dense(64)` architecture was retained, but the images were normalized to
`[0, 1]`, Adam used learning rate `1e-4`, and the maximum number of epochs was
increased to 50 with stricter early stopping.

The model stopped at epoch 17 and reached its best validation accuracy at
epoch 12.

| Metric | Result |
|---|---:|
| Mean Accuracy | 38.133% |
| Mean Recall | 19.467% |
| Mean Precision | 21.999% |

Training was more stable than in experiment 02, but the model still had a
strong bias towards `Building`. Several classes had zero recall.

### Experiment 04: baseline architecture with normalized inputs

This experiment returned to the original baseline architecture and changed
only the input scaling:

```text
Flatten -> ReLU -> Dense(13) -> Softmax
```

Images were normalized to `[0, 1]`; Adam, learning rate, batch size, maximum
epochs and original callbacks were kept as in experiment 01.

| Metric | Result |
|---|---:|
| Mean Accuracy | 41.280% |
| Mean Recall | 34.049% |
| Mean Precision | 36.760% |

The best validation model was obtained at epoch 20. All 13 classes obtained
non-zero recall, and the bias towards `Building` was reduced. This is currently
the best model and the reference point for future experiments.

## Current comparison

| Experiment | Main change | Mean Accuracy | Mean Recall | Mean Precision |
|---|---|---:|---:|---:|
| 01 | Original baseline | 38.560% | 30.950% | 31.433% |
| 02 | Added `Dense(64)` | 19.360% | 7.692% | 1.489% |
| 03 | `Dense(64)` + normalization + lower LR | 38.133% | 19.467% | 21.999% |
| 04 | Baseline + normalization | **41.280%** | **34.049%** | **36.760%** |

## Where to continue tomorrow

Use experiment 04 as the reference configuration. Do not add more dense layers
automatically: the results show that input scaling had a larger positive effect
than increasing network capacity.

Possible next experiments, changing one factor at a time:

1. Keep experiment 04 fixed and test Adam with learning rate `3e-4`.
2. Keep experiment 04 fixed and use a more gradual learning-rate reduction,
   such as `factor=0.5` and a longer patience.
3. Keep experiment 04 fixed and compare Adam with SGD plus momentum.
4. Test a different batch size, such as `32`.

For every new experiment, keep the same validation split and compare Mean
Accuracy first, with Mean Recall and Mean Precision as secondary diagnostics.
Do not use class weights if the immediate goal is only to maximize the
Codabench accuracy metric; they may improve minority-class recall while
reducing global accuracy.