# CNN Training Pipeline

## Overview

This project trains a 2-D Convolutional Neural Network (`LocalizationCNN`) to predict 3-D receiver positions (x, y, z) in metres from radio-frequency (RF) channel measurements collected in a pilot indoor-localization experiment (`pilot1`).

Two source files make up the pipeline:

| File | Purpose |
|---|---|
| `loc_cnn_pilot1.py` | Data loading, reshaping, model definition, training, evaluation |
| `loc_plotting.py` | All matplotlib visualisation helpers called by `loc_cnn_pilot1.py` |

---

## Directory Layout

```
Github/pilot1/
├── feature_datasets/
│   ├── pilot1_labels.csv           # shared ground-truth (x, y, z) positions
│   ├── pilot1_train_index.txt      # shared train split indices
│   ├── pilot1_test_index.txt       # shared test split indices
│   └── <group>/                    # one folder per dataset variant
│       ├── features_*.csv
│       ├── labels_*.csv
│       ├── train_index*.txt
│       └── test_index*.txt
├── localization_model/
│   ├── logs/
│   │   ├── ckpts/                      # model checkpoints (optional)
│   │   └── predictions/                # per-run prediction CSVs
│   ├── plots/                          # all generated figures
│   ├── loc_cnn_pilot1.py
│   └── loc_plotting.py
```

---

## Key Concepts

### Dimension Tuple

Every dataset is described by a tuple `(N_COMPLEX, N_TX, N_SYM, N_ANT, N_PKT)`:

| Field | Meaning |
|---|---|
| `N_COMPLEX` | Feature channels: `2` = real+imag, `1` = magnitude only, `3` = polar (r1, r2, Δφ) |
| `N_TX` | Number of transmitter positions (always 3) |
| `N_SYM` | Number of OFDM pilot symbols per packet |
| `N_ANT` | Number of receive antennas (1 or 2) |
| `N_PKT` | Number of packets / snapshots per measurement |

### CNN Input Shape

After reshaping, each sample becomes a `(C, H, W)` image fed to the CNN:

```
C = N_COMPLEX          (feature channels)
H = N_TX × N_ANT      (spatial height — transmitter/antenna axis)
W = N_SYM × N_PKT     (temporal width  — symbol/packet axis)
```

### Dataset Name Convention

The `dataset_name` string is the single source of truth that controls three things simultaneously:

- **Architecture** — which `model_param` and pooling sizes are selected in `LocalizationCNN.__init__`
- **Reshaping** — which axis reductions are applied in the `__main__` loop
- **Output paths** — where plots and CSVs are saved

The `_3dim` and `_2dim` suffixes signal progressive dimensionality reduction:

| Suffix | Meaning |
|---|---|
| *(none)* | Full resolution — all symbols and packets kept |
| `_3dim` | Symbol axis collapsed (single symbol selected or averaged); packets kept |
| `_2dim` | Both symbol and packet axes collapsed to a single value per TX/antenna |

---

## Adding a New Dataset

Follow these steps in order. All three locations must stay in sync because the name string drives branching in each one.

### Step 1 — Define the dimension tuple

```python
my_dims = (N_COMPLEX, N_TX, N_SYM, N_ANT, N_PKT)
```

### Step 2 — Create path variables

```python
group       = "pilot1_my_feature"
my_feat_dir = os.path.join(datadir, group, "features_my.csv")
my_lab_dir  = os.path.join(datadir, group, "labels_my.csv")
my_train_idx = os.path.join(datadir, group, "train_index_my.txt")
my_test_idx  = os.path.join(datadir, group, "test_index_my.txt")
```

### Step 3 — Create the registry tuple and add it to `dirs`

```python
my_dirs = ("pilot1_my_feature", my_feat_dir, my_lab_dir,
           my_train_idx, my_test_idx, my_dims)

dirs = [..., my_dirs]
```

### Step 4 — Add an architecture branch in `LocalizationCNN.__init__`

```python
elif dataset_name == "pilot1_my_feature":
    model_param = [256, 256, 512, 256]   # [conv_ch1, conv_ch2, fc1, fc2]
    # set ignore = True if the spatial map is too small for a second conv block
```

### Step 5 — Add a reshape branch in the `__main__` loop

Add an `elif` inside the `if N_COMPLEX != 3` block (or the `N_COMPLEX == 3` block if applicable):

```python
elif dataset_name == "pilot1_my_feature":
    # no reduction needed for the full-resolution variant
    pass
```

---

## Adding 3-D and 2-D Variants

### 3-D variant — collapse the symbol axis, keep packets

1. Add a second registry tuple with `"_3dim"` appended to the name, pointing to the **same CSVs and dim tuple** as the full-resolution version.
2. Add it to `dirs`.
3. In the reshape block, add:

```python
elif dataset_name == "pilot1_my_feature_3dim":
    sym_idx = 0                                    # pick first symbol, or:
    X = X[:, :, sym_idx:sym_idx+1, :, :]           # slice single symbol
    # alternatively: X = X.mean(axis=2, keepdims=True)  # average all symbols
    N_SYM = 1
```

4. Add an architecture branch:

```python
elif dataset_name == "pilot1_my_feature_3dim":
    model_param = [128, 128, 256, 128]
```

### 2-D variant — collapse both symbol and packet axes

1. Add a third registry tuple with `"_2dim"` appended, again pointing to the same CSVs.
2. Add it to `dirs`.
3. In the reshape block, add:

```python
elif dataset_name == "pilot1_my_feature_2dim":
    sym_idx = 0
    X = X[:, :, sym_idx:sym_idx+1, :, :]   # collapse symbol axis
    N_SYM = 1
    X = X.mean(axis=4, keepdims=True)       # collapse packet axis
    N_PKT = 1
```

4. The catch-all `elif dataset_name.find("2dim") != -1` in `LocalizationCNN.__init__` will automatically assign default 2-D params and set `ignore = True`. Only add an explicit branch if your variant needs non-default layer sizes or pooling.

---

## Training Configuration

Edit these variables at the top of `loc_cnn_pilot1.py` to change global training behaviour:

| Variable | Default | Description |
|---|---|---|
| `num_epoch` | `200` | Total training epochs per dataset/seed |
| `lr` | `0.001` | Initial learning rate for AdamW |
| `seed` range | `range(42, 43)` | Seeds to iterate over (extend to run multiple seeds) |

The optimiser uses AdamW with weight decay `1e-2`. The learning rate follows a Cosine Annealing Warm Restarts schedule (restart period `T_0=20` epochs, minimum LR `1e-6`).

---

## Output Files

| Path | Content |
|---|---|
| `logs/predictions/pred_<dataset>_<seed>.csv` | Per-sample predicted and true (x,y,z) plus Euclidean error |
| `logs/cnn_mean_errors.csv` | Summary table of mean Euclidean error per dataset and seed |
| `plots/<dataset>/train_val_overlay.png` | Training and validation loss curves |
| `plots/<dataset>/train_val_within_epoch_range.png` | Per-epoch batch loss variability with min/max bands |
| `plots/<dataset>/testerr.png` | Histogram of test Euclidean errors |
| `plots/<dataset>/seed_<s>/diff_<dataset>_<s>.png` | Scatter plot of predicted vs true positions |
| `plots/cnn_cdf_overlay.png` | CDF overlay across all dataset variants |

---

## `loc_plotting.py` — Plotting Functions Reference

All functions are imported via `from loc_plotting import *` in `loc_cnn_pilot1.py`. Each function creates its output directory automatically if it does not exist.

---

### `plot_error_histogram(errors, save_path, bins)`

Plots a histogram of per-sample Euclidean errors with a vertical dashed line marking the mean.

| Arg | Type | Default | Description |
|---|---|---|---|
| `errors` | `np.ndarray` | — | 1-D array of per-sample Euclidean errors in metres |
| `save_path` | `str` | `"error_histogram.png"` | Output file path |
| `bins` | `int` | `50` | Number of histogram bins |

**Output:** single PNG — error distribution with mean annotation.

---

### `plot_results(history_list, label_list, dataset_name, save_dir)`

Overlays training and validation loss curves for one or more training runs on a single axes. Training curves are solid lines; validation curves are dashed.

| Arg | Type | Default | Description |
|---|---|---|---|
| `history_list` | `list[dict]` | — | List of `hist` dicts returned by `train_model` |
| `label_list` | `list[str]` | — | Legend label for each run (e.g. `"lr=0.001_200ep"`) |
| `dataset_name` | `str` | — | Used in the plot title |
| `save_dir` | `str` | `"plots"` | Directory where `train_val_overlay.png` is written |

**Output:** `<save_dir>/train_val_overlay.png`

---

### `plot_epoch_variability(history_list, label_list, dataset_name, save_dir)`

Shows within-epoch batch loss variability across all runs. For each epoch (after an initial burn-in), all individual batch losses are plotted as translucent scatter points, and a mean ± min/max shaded band is drawn on top. Training is shown in blue; validation in green.

| Arg | Type | Default | Description |
|---|---|---|---|
| `history_list` | `list[dict]` | — | List of `hist` dicts returned by `train_model` |
| `label_list` | `list[str]` | — | Currently unused (reserved for multi-run labelling) |
| `dataset_name` | `str` | — | Used in the plot title |
| `save_dir` | `str` | `"plots"` | Directory where the plot is written |

`skip_epochs = 5` — the first 5 epochs are excluded to avoid the high-loss burn-in period distorting the y-axis scale.

**Output:** `<save_dir>/train_val_within_epoch_range.png`

---

### `plot_multiple_cdfs(error_dict, save_path)`

Overlays empirical CDFs of Euclidean error for multiple dataset variants on a single axes, with each variant as a separate coloured line.

| Arg | Type | Default | Description |
|---|---|---|---|
| `error_dict` | `dict[str, np.ndarray]` | — | Keys are run labels (e.g. `"pilot1_a_b_42"`); values are 1-D error arrays |
| `save_path` | `str` | `"cdf_overlay.png"` | Output file path |

**Output:** single PNG — CDF of localisation error in metres across all variants.

---

### `plot_mean_error_line(results_df, save_dir)`

Plots mean Euclidean error vs dataset name as a line chart, with one line per model variant. Intended for comparing baseline vs residual (or other) model types across datasets.

| Arg | Type | Default | Description |
|---|---|---|---|
| `results_df` | `pd.DataFrame` | — | Must have columns `"dataset"`, `"model"`, `"mean_error"` |
| `save_dir` | `str` | `"plots"` | Directory where `mean_error_line.png` is written |

**Note:** currently not called in `loc_cnn_pilot1.py` (the call is commented out at the end of `__main__`). Requires a `"model"` column not present in the default `mean_errors` list.

**Output:** `<save_dir>/mean_error_line.png`

---

### `plot_xy_prediction_lines(predictions, targets, train_xy, max_points, title, save_path)`

Scatter plot of predicted vs true (x, y) positions. An arrow is drawn from each prediction toward its true target, coloured by the XY error magnitude on a blue colormap (clipped at 10 m+). Training sample locations are shown as light grey background points.

| Arg | Type | Default | Description |
|---|---|---|---|
| `predictions` | `np.ndarray (N, 3)` | — | Predicted (x, y, z) positions |
| `targets` | `np.ndarray (N, 3)` | — | Ground-truth (x, y, z) positions |
| `train_xy` | `np.ndarray (M, 2 or 3)` | `None` | Training positions for background reference |
| `max_points` | `int or None` | `None` | Limit number of plotted samples (useful for large test sets) |
| `title` | `str` | `"XY Prediction vs Target"` | Plot title |
| `save_path` | `str` | `"xy_pred_lines.png"` | Output file path |

Arrow direction: tail at prediction, head at true target — so a longer arrow means a larger error. Colorbar ticks are in metres; the top tick label is shown as `"10+"` to indicate the colour saturates.

**Output:** single PNG at `save_path`.