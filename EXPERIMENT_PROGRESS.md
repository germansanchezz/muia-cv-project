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

### Experiment 05: larger ffNN and longer training

This experiment tested a deeper feedforward architecture with three hidden
layers:

```text
Flatten -> Dense(512) -> Dense(256) -> Dense(128) -> Dense(13) -> Softmax
```

It used Adam with learning rate `1e-4`, batch size `32`, and a maximum of 100
epochs. The original callback strategy was retained. Inputs were not
normalized, so this experiment remains comparable as an architecture and
training-parameter experiment within the ffNN challenge.

The best validation model was obtained at epoch 46 and training stopped at
epoch 86.

| Metric | Result |
|---|---:|
| Mean Accuracy | **47.573%** |
| Mean Recall | **40.211%** |
| Mean Precision | **42.761%** |

This is the best configuration tested so far. It improves on experiment 04,
although the gap between training and validation accuracy indicates some
overfitting. `Helipad` remains the weakest class, with zero recall.

### Experiment 06: experiment 05 with normalized inputs

Experiment 06 kept the experiment 05 architecture, optimizer, batch size, epoch
limit and callbacks, and changed only the input preprocessing by normalizing
training, validation and test images to `[0, 1]`.

Results:

| Metric | Result |
|---|---:|
| Mean Accuracy | **57.760%** |
| Mean Recall | **53.603%** |
| Mean Precision | **56.909%** |

The best validation model was obtained at epoch 51 and training stopped at
epoch 91. This is a strong exploratory result, but it is excluded from the
main challenge comparison because the current study is being kept focused on
architecture, optimizer, training parameters and stopping point. The
normalization option has consequently been removed from the common utilities.

Experiments 03, 04 and 06 used normalization and should be treated as
exploratory preprocessing experiments rather than the main comparison line.

## Current comparison

| Experiment | Main change | Mean Accuracy | Mean Recall | Mean Precision |
|---|---|---:|---:|---:|
| 01 | Original baseline | 38.560% | 30.950% | 31.433% |
| 02 | Added `Dense(64)` | 19.360% | 7.692% | 1.489% |
| 03 | `Dense(64)` + normalization + lower LR | 38.133% | 19.467% | 21.999% |
| 04 | Baseline + normalization | **41.280%** | **34.049%** | **36.760%** |
| 05 | Deeper ffNN, batch 32, longer training | **47.573%** | **40.211%** | **42.761%** |
| 06 | Experiment 05 + normalized inputs | **57.760%** | **53.603%** | **56.909%** |
| 07 | Experiment 05 + SGD with momentum | **49.920%** | **44.838%** | **49.124%** |
| 08 | SGD with momentum, learning rate 3e-4 | **52.480%** | **45.877%** | **46.808%** |
| 09 | Experiment 08 with batch size 16 | **55.733%** | **50.048%** | **53.474%** |

## Where to continue tomorrow

Use experiment 05 as the reference configuration for the main challenge line.
Experiment 06 is kept as an exploratory result, while experiment 07 tests the
optimization algorithm without normalization.

Possible next experiments, changing one factor at a time:

1. Treat experiment 09 as the current best non-normalized reference model.
2. Keep the best valid configuration fixed and test Adam with learning rate `3e-4`.
3. Keep the best valid configuration fixed and use a more gradual learning-rate reduction,
   such as `factor=0.5` and a longer patience.
4. Compare Adam with SGD plus momentum while changing no other factor.
5. Test a different batch size only after selecting the best optimizer and
    learning rate.

For every new experiment, keep the same validation split and compare Mean
Accuracy first, with Mean Recall and Mean Precision as secondary diagnostics.
Do not use class weights if the immediate goal is only to maximize the
Codabench accuracy metric; they may improve minority-class recall while
reducing global accuracy.