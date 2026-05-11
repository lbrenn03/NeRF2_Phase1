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

  | Dataset | Script                | Output Features      |
  |---------|-----------------------|----------------------|
  | Pilot1  | `flatten_channel_mul` | `features_chmul.csv` |

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

## Output CSVs

All CSVs are written to the directory specified by `--output_dir`. Labels are
always in meters (tiles × 0.6096).

| Script | File | Contents |
|--------|------|----------|
| `flatten_channel_mul` | `<features_file>` | Real/imag flattened features |
| `flatten_channel_mul` | `<labels_file>` | x, y, z in meters |

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

- Scripts create the output directory automatically if it does not exist.
- Labels are converted from tile units to meters by multiplying by `0.6096`.

---

## Dependencies

```
numpy
pandas
scipy
```