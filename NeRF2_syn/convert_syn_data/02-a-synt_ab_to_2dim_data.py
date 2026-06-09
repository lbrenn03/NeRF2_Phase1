"""
Convert synthetic_data_02/input → synthetic_data_02/output

Input data (from synthetic simulation):
  norml_syn_rx_pos.csv  : 1001 rows (header + 1000 data rows)
                          1000 rows = 500 positions × 2 antennas (row pairs)
                          Columns: x, y, z
                          ANT offset: ant1.x = center_x + 0.082
                                      ant2.x = center_x - 0.082

  synthetic_data.csv : 1001 rows (header + 1000 data rows)
                          Same pairing as labels (row 1=ant1, row 2=ant2, ...)
                          Columns: real_g01,imag_g01,real_g02,imag_g02,real_g03,imag_g03
                          g01/g02/g03 = TX0/TX1/TX2

Output (compatible with loc_cnn_Vivian.py channel_mul_nPck_2dim pipeline):
  labels.csv            : 500 rows, columns: x, y, z  (center position)
  features_channel.csv  : 500 rows, 12 columns
                          dims = (2, 3, 1, 2, 1) = (N_COMPLEX, N_TX, N_SYM, N_ANT, N_PKT)
                          N_FEAT = N_TX*N_SYM*N_ANT*N_PKT = 6
                          Column order (C-order reshape: TX slowest, ANT faster, PKT fastest):
                            feat_real_0  = TX0, ANT0 (ant1)
                            feat_real_1  = TX0, ANT1 (ant2)
                            feat_real_2  = TX1, ANT0
                            feat_real_3  = TX1, ANT1
                            feat_real_4  = TX2, ANT0
                            feat_real_5  = TX2, ANT1
                            feat_imag_0 .. feat_imag_5  (same order)
"""

import os
import numpy as np
import pandas as pd

ANT_OFFSET = 0.082  # meters — half-spacing between the two antennas along x

# # Absolute channel
# input_dir  = "synthetic_data_03/input"
# output_dir = "synthetic_data_03/output"
# a-b
input_dir = "group_synt_data/absolute_channel/input"
output_dir = "group_synt_data/absolute_channel/output"

os.makedirs(output_dir, exist_ok=True)

# --------------------------------------------------------------------------
# 1. Labels
# --------------------------------------------------------------------------
N_SAMPLES = 1000
labels_raw = pd.read_csv(os.path.join(input_dir, "norml_syn_rx_pos_500.csv"))
# assert len(labels_raw) == 2000, f"Expected 2000 label rows, got {len(labels_raw)}"

ant1 = labels_raw.iloc[0::2].reset_index(drop=True)  # rows 0,2,4,...  (ant1)
ant2 = labels_raw.iloc[1::2].reset_index(drop=True)  # rows 1,3,5,...  (ant2)

# Center x recovered from each antenna; verify they agree
center_from_ant1 = ant1["x"].values - ANT_OFFSET  # ant1.x = center + offset → center = ant1.x - offset

# Wait — check the actual relationship from the data:
# Row 1 (ant1): x=1.280, Row 2 (ant2): x=1.444
# If ant1.x + offset = ant2.x - offset → center would be between them
# Actually: t1 = ant1.x + 0.082, t2 = ant2.x - 0.082 (per task description)
# Let's verify:
t1 = ant1["x"].values + ANT_OFFSET
t2 = ant2["x"].values - ANT_OFFSET

max_discrepancy = np.max(np.abs(t1 - t2))
print(f"Max discrepancy between t1 and t2 across all 500 pairs: {max_discrepancy:.2e}")
assert max_discrepancy < 1e-9, (
    f"Antenna pair x-coordinates do not agree! Max diff = {max_discrepancy}"
)
print("Antenna pair check PASSED: t1 == t2 for all 500 positions")

center_x = t1  # = t2
center_y  = ant1["y"].values   # y is identical for both antennas
center_z  = ant1["z"].values   # z is identical for both antennas

labels_out = pd.DataFrame({"x": center_x, "y": center_y, "z": center_z})
labels_path = os.path.join(output_dir, "labels.csv")
labels_out.to_csv(labels_path, index=False)
print(f"Saved labels ({len(labels_out)} rows) → {labels_path}")

# --------------------------------------------------------------------------
# 2. Channel features
# --------------------------------------------------------------------------
channel_raw = pd.read_csv(os.path.join(input_dir, "channel_mul_synthetic_data.csv"))
# assert len(channel_raw) == 2000, f"Expected 2000 channel rows, got {len(channel_raw)}"

# Verify expected columns
expected_cols = ["real_g01", "imag_g01", "real_g02", "imag_g02", "real_g03", "imag_g03"]
assert list(channel_raw.columns) == expected_cols, (
    f"Unexpected columns: {list(channel_raw.columns)}"
)

ch_ant0 = channel_raw.iloc[0::2].reset_index(drop=True)  # ant0 (ant1 in 1-indexed)
ch_ant1 = channel_raw.iloc[1::2].reset_index(drop=True)  # ant1 (ant2 in 1-indexed)

# Build output in C-order for dims = (N_TX=3, N_SYM=1, N_ANT=2, N_PKT=1)
# Flat index: [TX0_ANT0, TX0_ANT1, TX1_ANT0, TX1_ANT1, TX2_ANT0, TX2_ANT1]
#             feat_real_0 .. feat_real_5  then  feat_imag_0 .. feat_imag_5
real_cols = {
    "feat_real_0": ch_ant0["real_g01"],  # TX0, ANT0
    "feat_real_1": ch_ant1["real_g01"],  # TX0, ANT1
    "feat_real_2": ch_ant0["real_g02"],  # TX1, ANT0
    "feat_real_3": ch_ant1["real_g02"],  # TX1, ANT1
    "feat_real_4": ch_ant0["real_g03"],  # TX2, ANT0
    "feat_real_5": ch_ant1["real_g03"],  # TX2, ANT1
}
imag_cols = {
    "feat_imag_0": ch_ant0["imag_g01"],  # TX0, ANT0
    "feat_imag_1": ch_ant1["imag_g01"],  # TX0, ANT1
    "feat_imag_2": ch_ant0["imag_g02"],  # TX1, ANT0
    "feat_imag_3": ch_ant1["imag_g02"],  # TX1, ANT1
    "feat_imag_4": ch_ant0["imag_g03"],  # TX2, ANT0
    "feat_imag_5": ch_ant1["imag_g03"],  # TX2, ANT1
}

features_out = pd.DataFrame({**real_cols, **imag_cols})
features_path = os.path.join(output_dir, "features_data.csv")
features_out.to_csv(features_path, index=False)
print(f"Saved features ({len(features_out)} rows × {len(features_out.columns)} cols) → {features_path}")

# --------------------------------------------------------------------------
# 3. Quick sanity check: reshape as loc_cnn_Vivian.py would for dims=(2,3,1,2,1)
# --------------------------------------------------------------------------
N_COMPLEX, N_TX, N_SYM, N_ANT, N_PKT = 2, 3, 1, 2, 1
N = len(features_out)
N_FEAT = N_TX * N_SYM * N_ANT * N_PKT  # = 6

X_real = features_out.iloc[:, :N_FEAT].values
X_imag = features_out.iloc[:, N_FEAT:].values
X = X_real + 1j * X_imag
X = X.reshape(N, N_TX, N_SYM, N_ANT, N_PKT)  # (500, 3, 1, 2, 1)

# Transpose and flatten to (N, N_TX*N_ANT, N_SYM*N_PKT) = (500, 6, 1)
X_t = X.transpose(0, 1, 3, 2, 4).reshape(N, N_TX * N_ANT, N_SYM * N_PKT)

# Stack real/imag → CNN input (500, 2, 6, 1)
inputs_cnn = np.stack([X_t.real, X_t.imag], axis=1)
print(f"\nSanity check — CNN input shape: {inputs_cnn.shape}  (expected: ({N}, 2, 6, 1))")
assert inputs_cnn.shape == (N, 2, 6, 1), "Shape mismatch!"
print("Shape check PASSED")
print("\nDone. Use dims=(2, 3, 1, 2, 1) when loading synthetic features in loc_cnn_Vivian.py.")
