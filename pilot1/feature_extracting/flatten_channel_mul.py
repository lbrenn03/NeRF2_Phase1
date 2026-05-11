"""
flatten_channel_mul.py
======================
Loads preprocessed .mat files, estimates per-packet channel responses via
cross-correlation and k-means clustering, then flattens the complex-valued
feature matrix into a 2D CSV for downstream ML pipelines.

Pipeline
--------
  .mat files  →  channel estimation (per antenna, per TX)
              →  k-means clustering (4 centroids per packet)
              →  feature matrix  X : (N, 3, 4, 2, target_packets) complex
                                 y : (N, 3) float  [rx_pos x/y/z]
              →  flatten + split real/imag
              →  features CSV  +  labels CSV

Output CSVs
-----------
  features_<name>.csv  —  columns: feat_real_0 … feat_real_N, feat_imag_0 … feat_imag_N
  labels_<name>.csv    —  columns: x, y, z

Usage
-----
  python flatten_channel_mul.py --data_dir ../data/Pilot1_Data_processed --output_dir ../feature_datasets --features_file features_chmul.csv --labels_file labels.csv

Arguments
---------
  --data_dir       Directory containing preprocessed .mat files
                   (default: ../Pilot1_Data_processed)
  --output_dir     Directory to write output CSVs; created if it doesn't exist
                   (default: ../Pilot1_Data_feature-csvs)
  --features_file  Filename for the features CSV
                   (default: features.csv)
  --labels_file    Filename for the labels CSV
                   (default: labels.csv)

Dependencies
------------
  numpy, pandas, scipy
"""

import numpy as np
import pandas as pd
import re
import os
from collections import defaultdict

import scipy.io
from scipy.cluster.vq import kmeans2
import warnings

TARGET_PACKETS = 8
USE_ANTENNAS = [1, 2]
SYM_STORE = 500        # symbols to keep per packet after trimming
SYM_OFFSET = 6         # drop first/last SYM_OFFSET symbols (filter edge transients)
# Source .mat files store all_syms_ant1/2 with shape (SYM_OFFSET + SYM_STORE + SYM_OFFSET, num_packets)
# i.e. at least 512 symbols per packet (= NUM_REF_SYMS from the receiver)

# ==============================================================================
# Reference signal — must match the TX/receiver pipeline exactly
# ==============================================================================
pn_bits = [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,1,1,1,1,1,1,0,1,1,1,0,0,0,1,0,0,1,1,1,1,1,0,0,0,1,1,0,0,1,1,1,1,1,0,1,0,1,1,0,0,1,0,1,1,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,1,1,0,0,1,0,0,1,1,0,1,0,0,0,0,1,0,0,1,0,1,0,1,0,0,0,0,1,1,1,1,0,1,0,1,1,1,0,1,0,1,1,0,1,1,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,1,1,0,0,1,1,0,0,0,0,1,0,1,0,1,1,0,1,0,1,1,1,0,0,0,1,1,0,1,1,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,0,0,1,1,1,1,0,1,1,0,1,1,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,1,0,1,1,0,1,0,1,0,1,0,0,0,1,1,1,1,1,0,1,1,1,1,0,0,1,0,0,1,0,1,1,0,0,0,0,0,1,0,0,1,1,0,0,1,0,0,0,1,0,1,0,0,0,1,1,0,1,1,0,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,1,1,1,0,1,1,1,1,1,1,1,0,0,1,0,0,0,0,1,1,0,0,0,1,0,1,1,0,1,1,1,0,1,0,0,0,0,1,1,0,1,0,1,0,1,1,0,0,1,1,1,1,0,0,1,0,1,1,0,1,1,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,1,1,0,0,0,0,0,0,1,0,1,1,0,0,0,1,0,1,0,0,1,1,1,0,1,1,0,0,1,1,1,0,0,0,1,0,1,1,1,1,1,1,0,1,0,1,0,0,0,1,0,1,1,1,0,1,1,0,1,0,1,1,0,0,0,0,1,1,0,0,1,1,0,1,1,0,1,0,1,0,0,0,0,0,1,1,1,0,1,0,0,1,1,1,1,0,1,0,0,1,1,0,1,0,1,0,0,1,0,0,1,1,1,0,0,0,0,0,1,1,1,1,1,0,0,1,1,1,0,0,1,1,0,1,1,1,1,0,1,0,0,0,1,0,1,0,1,0,1,1,0,1,1,1,1,1,0,0,0,0,1,0,0,1,1,1,0,1,0,0,0,1,1,1,0,1,0,1,1,1,1,1,0,1,1,0,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,1,0,0,1,0,1,0,1,1,0,0,0,1,1,1,0,0,1,1,1,1,1,1,1,0,1,1,0,0,0,0,1,0,0,0,1,1,0,1,0,0,1,1,1,0,0,1,0,0,1,1,1,1,0,0,0,0,1,1,0,1,1,1,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,0,1,1,1,1,1,0,1,0,0,1,0,0,1,0,1,0,0,0,0,0,0,1,1,0,1,0,0,0,1,1,0,0,1,0,1,1,1,0,1,0,0,1,0,1,1,0,1,0,0,0,1,0,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1,0,1,0,0,1,0,0,0,1,1,0,0,0,0,1,1,1,0,1,1,0,1,1,1,1,0,0,0,0,0,1,0,1,1,1,0,0,1,0,1,0,1,1,1,0,0,1,1,1,0,1,1,1,0,1,1,1,0,0,1,1,0,0,1,1,1,0,1,0,1,0,1,1,1,0,1,1,1,1,0,1,1,0,0,1,0,1,0,0,0,1,0,0,1,1,0,1,1,0,0,0,1,0,0,0,0,1,1,1,0,0,1,0,1,1,1,1,1,0,0,1,0,1,0,0,1,1,0,0,1,1,0,0,1,0,1,0,1,0,1,0,0,1,1,1,1,1,1,0,0,1,1,0,0,0,1,1,0,1,0,1,1,1,1,0,0,1,1,0,1,0,1,1,0,1,0,0,1,1,0,0,0,1,0,0,1,0,1,1,1,0,0,0,0,1,0,1,1,1,1,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,0,1,0,0,0,0,0,1,0,1,0,1,0,0,1,0,1,1,1,1,0,0,0,1,0,1,0,1,1,1,1,0,1,1,1,0,1,0,1,0,0,1,1,0,1,1,1,0,0,1,0,0,0,1,1,1,0,0,0]

_ref_bits = np.array(pn_bits, dtype=np.uint8)
if len(_ref_bits) % 2 != 0:
    _ref_bits = np.append(_ref_bits, 0)


def _qpsk_modulate(bits, phase_offset=np.pi / 4):
    """QPSK modulation matching MATLAB comm.QPSKModulator (BitInput=true, PhaseOffset=pi/4)."""
    assert len(bits) % 2 == 0
    b = bits.reshape(-1, 2)
    GRAY_MAP = {(0, 0): 0, (0, 1): 1, (1, 1): 2, (1, 0): 3}
    gray_idx = np.array([GRAY_MAP[tuple(bb)] for bb in b], dtype=int)
    return np.exp(1j * (phase_offset + np.pi / 2 * gray_idx))


# Pre-compute reference symbols once
REF_SYMS = _qpsk_modulate(_ref_bits)  # (512,) complex

print(f"Reference symbols shape: {REF_SYMS.shape}, first 5 symbols: {REF_SYMS[:5]}")

# assert False


def compute_channel_h(all_syms):
    """
    Aligns all_syms against REF_SYMS using cross-correlation to find the delay,
    then estimates the channel by multiplying with the conjugate of the reference.

    Input:  all_syms  (500, num_packets) complex
    Output: channel_h (488, num_packets) complex  (due to SYM_OFFSET trimming)
    """
    n_syms, num_packets = all_syms.shape   # 500, P
    n_ref = len(REF_SYMS)                  # 512
    
    # Pre-allocate output array with the exact trimmed shape (500 - 2*6 = 488)
    out_len = n_syms - 2 * SYM_OFFSET
    channel_h = np.zeros((out_len, num_packets), dtype=complex)

    for p in range(num_packets):
        # 1. Robust alignment via cross-correlation magnitude
        # np.correlate with mode='valid' slides the 500-symbol received signal
        # across the 512-symbol reference. The absolute value ignores phase rotation.
        corr = np.correlate(REF_SYMS, all_syms[:, p], mode='valid')
        best_start = np.argmax(np.abs(corr))
        
        # print(f'---> Packet {p}: best_start={best_start}, corr_peak={corr[best_start]}, corr_mag={np.abs(corr[best_start])}')
        assert best_start == 0, "We want this data match the flatten_channel_mul.py"
        
        # 2. Extract the perfectly aligned reference sequence for THIS packet
        trimmed_ref = REF_SYMS[best_start : best_start + n_syms]
        
        # 3. Channel estimation: Y * X*
        # Valid because QPSK reference symbols have unit power (|X|^2 = 1)
        h_est = all_syms[:, p] * trimmed_ref.conj()
        
        # 4. Trim edge transients and store in the final array
        channel_h[:, p] = h_est[SYM_OFFSET:-SYM_OFFSET]

    return channel_h

def kmeans_cluster_means(syms, filepath='unknown', packet=0, ant=0):
    """K-means (k=4) on channel-h symbols; returns 4 complex centroids."""
    pts = np.column_stack([syms.real, syms.imag])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        centroids, labels = kmeans2(pts, 4, iter=20, minit='++', missing='warn')
        if w:
            means = centroids[:, 0] + 1j * centroids[:, 1]
            print(f"  kmeans warning ant{ant} packet {packet}: {filepath}  — {w[0].message}")
            print(f"    centroids: {means}")
    return centroids[:, 0] + 1j * centroids[:, 1]


def compute_cluster_means(channel_h, filepath='unknown'):
    """
    Run k-means per packet on channel_h to get (4, num_packets) complex centroids.

    Input:  channel_h  (500, num_packets)
    Output: means      (4,   num_packets)
    """
    num_packets = channel_h.shape[1]
    means = np.full((4, num_packets), np.nan, dtype=complex)
    for p in range(num_packets):
        try:
            means[:, p] = kmeans_cluster_means(channel_h[:, p], filepath=filepath, packet=p)
        except Exception as e:
            print(f"    kmeans failed for packet {p}: {e}", flush=True)
    return means


def normalize_packet(data):
    num_packets = int(data['num_packets'].flatten()[0])

    for key in ['cluster_means_channel_ant1', 'cluster_means_channel_ant2']:
        array = data[key]

        if num_packets > TARGET_PACKETS:
            data[key] = array[:, :TARGET_PACKETS]  # Trim to TARGET_PACKETS
        elif num_packets < TARGET_PACKETS:
            padding = TARGET_PACKETS - num_packets
            data[key] = np.pad(array, ((0, 0), (0, padding)))  # pad with zeros

    data['num_packets'] = TARGET_PACKETS
    return data


def load_mat_file(filepath):
    # try:
        mat_data = scipy.io.loadmat(filepath)
        final_data = mat_data['final_data'][0, 0]

        data = {
            'num_packets':   final_data['num_packets'],   # (1, num_packets)
            'all_syms_ant1': final_data['all_syms_ant1'], # (512, num_packets)
            'all_syms_ant2': final_data['all_syms_ant2'], # (512, num_packets)
            'rx_pos': final_data['rx_pos'].flatten(),
            'tx_pos': final_data['tx_pos'].flatten(),
        }

        # Step 1: compute per-symbol channel estimate (500 symbols after trim)
        channel_h_ant1 = compute_channel_h(data['all_syms_ant1'])
        # print(f'data["all_syms_ant1"] shape: {data["all_syms_ant1"].shape}')
        # print(f'--> shape: {channel_h_ant1.shape}')
        # assert False
        channel_h_ant2 = compute_channel_h(data['all_syms_ant2'])

        # Step 2: k-means → 4 centroids per packet (same approach as flatten.py)
        data['cluster_means_channel_ant1'] = compute_cluster_means(channel_h_ant1, filepath=filepath)
        data['cluster_means_channel_ant2'] = compute_cluster_means(channel_h_ant2, filepath=filepath)

        return normalize_packet(data)

    # except Exception as e:
    #     print(f"Error loading {filepath}: {e}")
    #     return None


def parse_filename(filepath):
    match = re.search(r'tx(\d+)_(n?\d+_\d+)\.mat', os.path.basename(filepath))
    if match:
        tx_id = int(match.group(1))
        rx_id = match.group(2)
        return tx_id, rx_id
    return None, None


def group_files(data_dir):
    files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.mat')])

    groups = defaultdict(dict)
    for f in files:
        tx_id, rx_id = parse_filename(f)
        if tx_id is not None:
            groups[rx_id][tx_id] = f

    for rx_id, tx_files in groups.items():
        if len(tx_files) != 3:
            print(f"rx_id={rx_id} has {len(tx_files)}/3 TX files: {list(tx_files.keys())}")

    return groups


def build_feature_matrix(groups):
    """
    Returns:
        X : (N, 3, 4, 2, target_packets) complex   — same shape as flatten.py
        y : (N, 3) float  [rx_pos]
    """
    X, y = [], []

    for rx_id in sorted(groups.keys()):
        tx_files = groups[rx_id]

        if len(tx_files) != 3:
            print(f"Skipping rx_id={rx_id}, missing TX files")
            continue

        tx_features = []
        rx_pos = None

        for tx_id in sorted(tx_files.keys()):
            data = load_mat_file(tx_files[tx_id])
            if data is None:
                break

            ant_data = []
            if 1 in USE_ANTENNAS:
                ant_data.append(data['cluster_means_channel_ant1'])  # (4, target_packets)
            if 2 in USE_ANTENNAS:
                ant_data.append(data['cluster_means_channel_ant2'])  # (4, target_packets)
            ch = np.stack(ant_data, axis=1)  # (4, 2, target_packets)

            tx_features.append(ch)
            rx_pos = data['rx_pos']

        if len(tx_features) != 3:
            print(f"Skipping rx_id={rx_id}, failed to load all TX files")
            continue

        X.append(np.stack(tx_features, axis=0))  # (3, 4, 2, target_packets)
        y.append(rx_pos)

    if not X:
        return np.empty((0, 3, 4, len(USE_ANTENNAS), TARGET_PACKETS), dtype=complex), np.empty((0, 3))
    return np.array(X), np.array(y)  # (N, 3, 4, 2, target_packets), (N, 3)

def flatten_and_save(X, y, output_dir, features_file, labels_file):
    """
    Splits complex values into real/imag, flattens to 2D and saves as CSV.
    features_channel.csv columns: feat_real_0 ... feat_real_N, feat_imag_0 ... feat_imag_N
    labels_channel.csv columns: x, y, z
    """
    N = X.shape[0]
    flat = X.reshape(N, -1)  # (N, 3*500*2*target_packets) complex
    n_feat = flat.shape[1]
    real_names = [f'feat_real_{i}' for i in range(n_feat)]
    imag_names = [f'feat_imag_{i}' for i in range(n_feat)]
    df_X = pd.DataFrame(
        np.hstack([flat.real, flat.imag]),
        columns=real_names + imag_names
    )
    df_y = pd.DataFrame(y, columns=['x', 'y', 'z'])

    os.makedirs(output_dir, exist_ok=True)
    features_path = os.path.join(output_dir, features_file)
    labels_path = os.path.join(output_dir, labels_file)

    df_X.to_csv(features_path, index=False)
    df_y.to_csv(labels_path, index=False)
    print(f"Saved X: {df_X.shape} -> {features_path}")
    print(f"Saved y: {df_y.shape} -> {labels_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='Directory containing preprocessed .mat files',
                        default='../data/Pilot1_Data_processed', required=True)
    parser.add_argument('--output_dir', help='Directory to save output CSV files',
                        default='../feature_datasets', required=False)
    parser.add_argument('--features_file', help='Filename for the features CSV',
                        default='features_chmul.csv', required=False)
    parser.add_argument('--labels_file', help='Filename for the labels CSV',
                        default='labels.csv', required=False)
    args = parser.parse_args()

    groups = group_files(args.data_dir)
    X, y = build_feature_matrix(groups)
    print(f"NaN count in X: {np.isnan(X.real).sum()}")
    print(f"X shape : {X.shape}")   # (N, 3, 4, 2, 8)
    print(f"y shape : {y.shape}")   # (N, 3)
    flatten_and_save(X, y, args.output_dir, args.features_file, args.labels_file)
