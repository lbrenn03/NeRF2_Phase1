# Feature Extraction for NeRF2 Training Data

This directory contains scripts to load preprocessed MIMO `.mat` files,
estimate channel responses, and save flattened feature matrices as CSVs
for downstream ML pipelines.

---

## Directory Contents

### Scripts

#### `generate_features.sh`
- **Purpose**: Runs all feature extraction scripts across both F1 and F2
  datasets, generating the full set of real/imag and polar feature CSVs
  in one shot.
- **Usage**:
```bash
  ./generate_features.sh
```
- **What it runs**:

  | Dataset | Script | Mode | Output Features |
  |---------|--------|------|-----------------|
  | F1 train | `flatten_channel_mul` | — | `f1train_features_ch_mul.csv` |
  | F1 test  | `flatten_channel_mul` | — | `f1test_features_ch_mul.csv` |
  | F2 train | `flatten_channel_mul` | — | `f2train_features_ch_mul.csv` |
  | F2 test  | `flatten_channel_mul` | — | `f2test_features_ch_mul.csv` |
  | F1 train+test | `flatten_rphi` | `polar` | `f1train/test_features_rphi.csv` |
  | F1 train+test | `flatten_rphi` | `polar_relative` | `f1train/test_features_rdphi.csv` |
  | F2 train+test | `flatten_rphi` | `polar` | `f2train/test_features_rphi.csv` |
  | F2 train+test | `flatten_rphi` | `polar_relative` | `f2train/test_features_rdphi.csv` |

---

#### `pilot2_flatten_channel_mul.py`
- **Purpose**: Extracts raw complex-valued channel features. For each `.mat`
  file, estimates the channel response per antenna and TX using cross-correlation
  against dual PN reference sequences (S1/S2), then compresses each packet's
  symbols into 4 k-means centroids. The result is split into real and imaginary
  components and saved as a flat 2D CSV.
- **Output Shape**: `(N, 4, 6, 2, 4, 8)` → flattened to `(N, 3072)` after real/imag split.
- **Output Columns**: `feat_real_0 ... feat_real_1535, feat_imag_0 ... feat_imag_1535`
- **Usage**:
```bash
  # Defaults
  python pilot2_flatten_channel_mul.py

  # Custom paths
  python pilot2_flatten_channel_mul.py \
      --data_dir      ../F1_MIMO_train_processed \
      --output_dir    ../feature_datasets \
      --features_file features_mul.csv \
      --labels_file   labels_mul.csv
```
- **Arguments**:

  | Argument | Default | Description |
  |----------|---------|-------------|
  | `--data_dir` | `../F1_MIMO_train_processed` | Directory containing `.mat` files |
  | `--output_dir` | `output` | Directory to write CSVs (created if missing) |
  | `--features_file` | `features.csv` | Filename for features CSV |
  | `--labels_file` | `labels.csv` | Filename for labels CSV |

---

#### `pilot2_flatten_rphi.py`
- **Purpose**: Same channel estimation pipeline as `flatten_channel_mul`, but
  converts the complex features into polar form before saving. Supports four
  normalization modes selected via `--mode` (required).
- **Usage**:
```bash
  # F1 polar (r, phi)
  python pilot2_flatten_rphi.py --mode polar \
      --train_dir      ../data/F1_MIMO_train_processed \
      --test_dir       ../data/F1_MIMO_test_processed \
      --train_features f1train_features_rphi.csv \
      --test_features  f1test_features_rphi.csv

  # F1 relative-phase (r1, r2, dphi)
  python pilot2_flatten_rphi.py --mode polar_relative \
      --train_dir      ../data/F1_MIMO_train_processed \
      --test_dir       ../data/F1_MIMO_test_processed \
      --train_features f1train_features_rdphi.csv \
      --test_features  f1test_features_rdphi.csv

  # F2 polar
  python pilot2_flatten_rphi.py --mode polar \
      --train_dir      ../data/F2_MIMO_train_processed \
      --test_dir       ../data/F2_MIMO_test_processed \
      --train_features f2train_features_rphi.csv \
      --test_features  f2test_features_rphi.csv

  # F2 relative-phase
  python pilot2_flatten_rphi.py --mode polar_relative \
      --train_dir      ../data/F2_MIMO_train_processed \
      --test_dir       ../data/F2_MIMO_test_processed \
      --train_features f2train_features_rdphi.csv \
      --test_features  f2test_features_rdphi.csv
```
- **Arguments**:

  | Argument | Default | Description |
  |----------|---------|-------------|
  | `--train_dir` | `../data/F1_MIMO_train_processed` | Train `.mat` files directory |
  | `--test_dir` | `../data/F1_MIMO_test_processed` | Test `.mat` files directory |
  | `--output_dir` | `../feature_datasets` | Directory to write CSVs (created if missing) |
  | `--mode` | *(required)* | See modes table below |
  | `--train_features` | `train_features.csv` | Train features filename |
  | `--train_labels` | `train_labels.csv` | Train labels filename |
  | `--test_features` | `test_features.csv` | Test features filename |
  | `--test_labels` | `test_labels.csv` | Test labels filename |

- **Flatten Modes**:

  | Mode | Normalization | Antenna Collapse | # Features |
  |------|--------------|-----------------|------------|
  | `polar` | Global r_max from train set | None — per-element (r, phi) | 3072 |
  | `polar_relative` | Global r_max from train set | Antenna pair → (r1, r2, dphi) | 2304 |
  | `polar_samplenorm` | Per-sample r_max | None — per-element (r, phi) | 3072 |
  | `polar_relative_samplenorm` | Per-sample r_max | Antenna pair → (r1, r2, dphi) | 2304 |

  For `polar` and `polar_relative`, r_max is computed from the train set and
  applied to both splits. This script always processes train and test together.

---

## Output CSVs

All CSVs are written to the directory specified by `--output_dir`. Labels are
always in meters (tiles × 0.6096).

| Script | File | Contents |
|--------|------|----------|
| `flatten_channel_mul` | `<features_file>` | Real/imag flattened features |
| `flatten_channel_mul` | `<labels_file>` | x, y, z in meters |
| `flatten_rphi` | `<train_features>`, `<test_features>` | Polar features |
| `flatten_rphi` | `<train_labels>`, `<test_labels>` | x, y, z in meters |

---

## Workflow Overview

1. **Prepare Input Data**:
   - Ensure preprocessed `.mat` files are available in the appropriate
     `train` and `test` directories. These are produced by the signal
     processing scripts in the parent directory.

2. **Choose a Script**:
   - Run `generate_features.sh` to generate all feature sets at once, or
     run the individual scripts manually for a specific dataset or mode.

3. **Run Feature Extraction**:
   - Specify input directories, output directory, and output filenames
     via command-line arguments.

4. **Inspect Outputs**:
   - Feature CSVs will contain one row per sample with all features flattened.
   - Label CSVs will contain `x`, `y`, `z` columns in meters.

---

## Notes

- Both scripts create the output directory automatically if it does not exist.
- For `polar` and `polar_relative` modes, always run train and test extractions
  together in the same call so that the same r_max is applied to both splits.
- Labels are converted from tile units to meters by multiplying by `0.6096`.

---

## Dependencies

```
numpy
pandas
scipy
```