"""
pilot2_flatten_polar.py
=======================
Loads MIMO .mat files with dual-source (S1/S2) transmissions, builds the
feature matrix, then flattens the complex-valued channel estimates into
polar-coordinate CSV representations for downstream ML pipelines.

Pipeline
--------
  .mat files  →  dual-source channel estimation (S1/S2, per antenna, per TX)
              →  k-means clustering (4 centroids per packet, per source)
              →  feature matrix  X : (N, 4, 6, 2, 4, target_packets) complex
                                     (angle, TX×source, antenna, cluster, packet)
                                 y : (N, 3) float  [x, y, z] in meters
              →  polar decomposition + normalization
              →  features CSV  +  labels CSV

Flatten Modes  (selected via --mode)
-------------------------------------
  polar_relative        Per-sample magnitude normalization. For each
                        (angle, tx, cluster, packet) cell, collapses the two
                        antennas into (r1, r2, dphi = phi2-phi1). r_max is
                        computed independently per sample.
                        Output: 4*6*4*8*3 = 2304 features
                        Columns: feat_r1_0..767, feat_r2_0..767, feat_dphi_0..767

  polar_1norm           Global magnitude normalization using r_max from the
                        train set. Keeps real/imag flattened, converts to
                        (r, phi) polar form. r normalized to [0,1] globally;
                        phi mapped from [-pi,pi] to [0,1].
                        Output: 2 * 4*6*2*4*8 = 3072 features
                        Columns: feat_r_0..1535, feat_phi_0..1535

  polar_relative_1norm  Same antenna-pair relative-phase layout as
                        polar_relative, but r_max is derived globally from
                        the train set rather than per sample.
                        Output: 4*6*4*8*3 = 2304 features
                        Columns: feat_r1_0..767, feat_r2_0..767, feat_dphi_0..767

Output CSVs
-----------
  <train_features>  —  polar features for training split
  <train_labels>    —  columns: x, y, z  (train)
  <test_features>   —  polar features for test split
  <test_labels>     —  columns: x, y, z  (test)

  For modes polar_1norm and polar_relative_1norm, r_max is always computed
  from the train split and applied to both train and test.

Usage
-----
python pilot2_flatten_polar.py --mode polar_relative --train_dir ../../../Pilot2_MIMO/F1_MIMO_train_processed --test_dir ../../../Pilot2_MIMO/F1_MIMO_test_processed --train_features train_features_rdphi.csv --test_features test_features_rdphi.csv

python pilot2_flatten_polar.py --mode polar_1norm --train_dir ../../../Pilot2_MIMO/F1_MIMO_train_processed --test_dir ../../../Pilot2_MIMO/F1_MIMO_test_processed --train_features train_features_rphi.csv --test_features test_features_rphi.csv

python pilot2_flatten_polar.py --mode polar_relative_1norm --train_dir ../../../Pilot2_MIMO/F1_MIMO_train_processed --test_dir ../../../Pilot2_MIMO/F1_MIMO_test_processed --train_features train_features_rel1norm.csv --test_features test_features_rel1norm.csv

Arguments
---------
  --train_dir       Directory containing train .mat files
                    (default: ../F1_MIMO_train_processed)
  --test_dir        Directory containing test .mat files
                    (default: ../F1_MIMO_test_processed)
  --output_dir      Directory to write output CSVs; created if it doesn't exist
                    (default: output)
  --mode            Flatten function to run: polar_relative | polar_1norm |
                    polar_relative_1norm  (default: polar_relative)
  --train_features  Filename for the train features CSV  (default: train_features.csv)
  --train_labels    Filename for the train labels CSV    (default: train_labels.csv)
  --test_features   Filename for the test features CSV   (default: test_features.csv)
  --test_labels     Filename for the test labels CSV     (default: test_labels.csv)

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
SYM_STORE = 500
SYM_OFFSET = 6

# ==============================================================================
# Reference signal — must match the TX/receiver pipeline exactly
# ==============================================================================
pn_bits1 = [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,1,1,1,1,1,1,0,1,1,1,0,0,0,1,0,0,1,1,1,1,1,0,0,0,1,1,0,0,1,1,1,1,1,0,1,0,1,1,0,0,1,0,1,1,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,1,1,0,0,1,0,0,1,1,0,1,0,0,0,0,1,0,0,1,0,1,0,1,0,0,0,0,1,1,1,1,0,1,0,1,1,1,0,1,0,1,1,0,1,1,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,1,1,0,0,1,1,0,0,0,0,1,0,1,0,1,1,0,1,0,1,1,1,0,0,0,1,1,0,1,1,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,0,0,1,1,1,1,0,1,1,0,1,1,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,1,0,1,1,0,1,0,1,0,1,0,0,0,1,1,1,1,1,0,1,1,1,1,0,0,1,0,0,1,0,1,1,0,0,0,0,0,1,0,0,1,1,0,0,1,0,0,0,1,0,1,0,0,0,1,1,0,1,1,0,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,1,1,1,0,1,1,1,1,1,1,1,0,0,1,0,0,0,0,1,1,0,0,0,1,0,1,1,0,1,1,1,0,1,0,0,0,0,1,1,0,1,0,1,0,1,1,0,0,1,1,1,1,0,0,1,0,1,1,0,1,1,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,1,1,0,0,0,0,0,0,1,0,1,1,0,0,0,1,0,1,0,0,1,1,1,0,1,1,0,0,1,1,1,0,0,0,1,0,1,1,1,1,1,1,0,1,0,1,0,0,0,1,0,1,1,1,0,1,1,0,1,0,1,1,0,0,0,0,1,1,0,0,1,1,0,1,1,0,1,0,1,0,0,0,0,0,1,1,1,0,1,0,0,1,1,1,1,0,1,0,0,1,1,0,1,0,1,0,0,1,0,0,1,1,1,0,0,0,0,0,1,1,1,1,1,0,0,1,1,1,0,0,1,1,0,1,1,1,1,0,1,0,0,0,1,0,1,0,1,0,1,1,0,1,1,1,1,1,0,0,0,0,1,0,0,1,1,1,0,1,0,0,0,1,1,1,0,1,0,1,1,1,1,1,0,1,1,0,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,1,0,0,1,0,1,0,1,1,0,0,0,1,1,1,0,0,1,1,1,1,1,1,1,0,1,1,0,0,0,0,1,0,0,0,1,1,0,1,0,0,1,1,1,0,0,1,0,0,1,1,1,1,0,0,0,0,1,1,0,1,1,1,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,0,1,1,1,1,1,0,1,0,0,1,0,0,1,0,1,0,0,0,0,0,0,1,1,0,1,0,0,0,1,1,0,0,1,0,1,1,1,0,1,0,0,1,0,1,1,0,1,0,0,0,1,0,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1,0,1,0,0,1,0,0,0,1,1,0,0,0,0,1,1,1,0,1,1,0,1,1,1,1,0,0,0,0,0,1,0,1,1,1,0,0,1,0,1,0,1,1,1,0,0,1,1,1,0,1,1,1,0,1,1,1,0,0,1,1,0,0,1,1,1,0,1,0,1,0,1,1,1,0,1,1,1,1,0,1,1,0,0,1,0,1,0,0,0,1,0,0,1,1,0,1,1,0,0,0,1,0,0,0,0,1,1,1,0,0,1,0,1,1,1,1,1,0,0,1,0,1,0,0,1,1,0,0,1,1,0,0,1,0,1,0,1,0,1,0,0,1,1,1,1,1,1,0,0,1,1,0,0,0,1,1,0,1,0,1,1,1,1,0,0,1,1,0,1,0,1,1,0,1,0,0,1,1,0,0,0,1,0,0,1,0,1,1,1,0,0,0,0,1,0,1,1,1,1,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,0,1,0,0,0,0,0,1,0,1,0,1,0,0,1,0,1,1,1,1,0,0,0,1,0,1,0,1,1,1,1,0,1,1,1,0,1,0,1,0,0,1,1,0,1,1,1,0,0,1,0,0,0,1,1,1,0,0,0,1]
pn_bits2 = [0,1,1,0,0,1,1,0,1,0,1,0,1,0,0,1,0,1,0,0,1,0,0,1,0,1,1,0,1,1,1,0,1,0,1,0,1,0,0,0,0,1,1,0,1,0,1,1,1,0,1,0,0,1,1,0,0,0,1,1,1,0,1,0,0,1,1,1,0,0,0,1,1,0,0,0,1,0,1,1,1,0,0,1,0,0,0,1,1,0,0,1,1,0,0,0,1,0,1,1,1,0,1,1,0,1,0,1,0,0,0,1,0,0,0,0,1,0,1,1,1,0,0,0,0,0,1,1,1,0,1,1,0,1,0,0,0,0,1,1,0,0,1,0,0,1,1,1,0,0,0,0,1,0,1,0,1,0,0,1,0,1,0,1,1,0,0,1,0,0,0,0,1,0,1,0,1,0,1,0,0,0,0,1,0,1,1,1,1,1,0,0,1,0,1,0,0,0,0,0,0,1,1,0,1,0,1,0,1,0,0,0,0,1,0,0,1,1,1,1,0,0,1,0,1,1,1,0,0,0,1,1,1,0,0,1,0,0,0,1,1,0,1,1,1,1,0,0,0,0,1,1,0,0,1,1,0,1,0,1,0,0,1,0,0,1,1,0,0,0,0,1,1,1,0,0,1,0,1,1,0,0,0,1,0,0,0,0,1,0,0,0,1,1,0,0,1,1,0,1,0,1,1,1,0,1,0,0,1,0,0,1,0,1,0,1,1,0,1,0,0,1,0,0,0,1,0,0,0,1,1,0,0,0,0,1,1,1,0,1,1,0,1,1,0,1,1,1,0,1,0,0,1,0,0,0,0,1,0,1,1,1,0,0,0,0,0,0,0,1,0,0,1,1,1,1,0,0,0,0,1,0,1,0,1,0,1,1,0,0,0,1,0,0,0,1,1,0,0,0,1,0,1,0,1,0,0,1,0,1,1,1,1,1,0,1,1,0,0,0,0,0,1,0,1,0,1,0,0,0,1,0,0,0,0,1,0,1,1,0,1,1,1,0,1,0,0,0,1,1,0,0,1,1,1,0,0,0,1,1,1,0,1,0,0,1,0,1,0,1,0,1,0,0,0,0,0,0,1,1,1,0,0,1,0,0,1,0,1,1,1,1,0,0,1,0,1,1,1,1,0,0,1,1,1,1,1,1,0,1,0,1,1,0,1,1,0,1,0,1,0,0,1,0,0,1,0,0,0,1,0,1,0,1,0,0,0,0,1,1,0,0,0,1,0,0,1,1,1,1,1,0,0,1,1,0,1,0,0,1,1,1,1,1,1,1,1,0,1,1,0,1,1,1,1,0,0,1,0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,0,0,1,1,1,0,1,1,0,0,0,0,0,1,1,1,0,1,0,0,1,0,0,0,1,0,1,0,1,1,0,0,1,1,1,1,0,0,1,1,0,1,1,1,1,1,0,0,1,0,1,1,1,1,1,0,1,0,1,1,0,1,0,0,1,0,1,0,1,0,0,0,0,0,0,0,1,0,1,1,1,0,1,0,0,1,0,0,1,1,0,0,0,1,1,1,0,0,0,0,0,1,1,1,0,0,1,0,0,0,0,1,0,1,1,1,0,1,0,0,1,1,1,0,0,1,1,1,1,0,0,0,0,1,1,0,0,0,1,0,0,0,1,1,0,1,0,1,1,1,0,0,0,0,1,1,1,1,1,1,1,0,1,1,0,1,1,0,1,0,0,0,0,0,0,0,1,1,0,1,0,0,1,1,1,0,0,1,0,1,0,0,0,1,1,0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,1,0,1,0,1,0,0,1,1,0,1,1,1,0,1,1,1,1,1,1,0,1,1,1,1,1,0,0,0,1,0,0,0,1,1,1,0,1,0,1,1,1,0,1,1,0,1,0,0,0,0,0,1,0,1,1,0,1,1,1,1,1,0,0,0,1,0,1,0,1,0,1,0,0,0,1,1,0,0,1,1,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,1,1,1,0,0,0,0,1,0,0,0,0,0,0,1,1,1,1,0,0,0,1,0,1,0,1,0,0,0,0,1,1,1,1,0,1,1,1,1,0,0,0,0,1,0,0,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,0,0,0,1,1,0,1,1,1,0,0,0,0,0,1,0,1,1,0,1,1,0,1,1,1,0,0,0,0,1,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,1,0,1,0,0,0,0,1,0,0,1,1,0,1]

_ref_bits1 = np.array(pn_bits1, dtype=np.uint8)
if len(_ref_bits1) % 2 != 0:
    _ref_bits1 = np.append(_ref_bits1, 0)
_ref_bits2 = np.array(pn_bits2, dtype=np.uint8)
if len(_ref_bits2) % 2 != 0:
    _ref_bits2 = np.append(_ref_bits2, 0)

def _qpsk_modulate(bits, phase_offset=np.pi / 4):
    """QPSK modulation matching MATLAB comm.QPSKModulator (BitInput=true, PhaseOffset=pi/4)."""
    assert len(bits) % 2 == 0
    b = bits.reshape(-1, 2)
    GRAY_MAP = {(0, 0): 0, (0, 1): 1, (1, 1): 2, (1, 0): 3}
    gray_idx = np.array([GRAY_MAP[tuple(bb)] for bb in b], dtype=int)
    return np.exp(1j * (phase_offset + np.pi / 2 * gray_idx))


# Pre-compute reference symbols once
REF_SYM1 = _qpsk_modulate(_ref_bits1)  # (512,) complex
REF_SYM2 = _qpsk_modulate(_ref_bits2)

print(f"Reference symbols 1 shape: {REF_SYM1.shape}, first 5 symbols: {REF_SYM1[:5]}")
print(f"Reference symbols 2 shape: {REF_SYM2.shape}, first 5 symbols: {REF_SYM2[:5]}")

def compute_channel_h(all_syms):
    """
    Aligns all_syms against REF_SYMS using cross-correlation to find the delay,
    then estimates the channel by multiplying with the conjugate of the reference.

    Input:  all_syms  (500, num_packets) complex
    Output: channel_h (488, num_packets) complex  (due to SYM_OFFSET trimming)
    """
    n_syms, num_packets = all_syms.shape
    n_ref = len(REF_SYMS)

    out_len = n_syms - 2 * SYM_OFFSET
    channel_h = np.zeros((out_len, num_packets), dtype=complex)

    for p in range(num_packets):
        corr = np.correlate(REF_SYMS, all_syms[:, p], mode='valid')
        best_start = np.argmax(np.abs(corr))

        if best_start + n_syms > len(REF_SYMS):
            raise ValueError(
                f"Alignment overflow: start={best_start}, n_syms={n_syms}, ref_len={len(REF_SYMS)}"
            )

        trimmed_ref = REF_SYMS[best_start : best_start + n_syms]
        h_est = all_syms[:, p] * trimmed_ref.conj()
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
        syms = channel_h[:, p]

        if syms is None or len(syms) == 0:
            means[:, p] = 0
            continue

        syms = np.asarray(syms)

        if syms.size < 10 or np.allclose(syms, 0):
            means[:, p] = 0
            continue

        if np.isnan(syms).any():
            means[:, p] = 0
            continue

        try:
            means[:, p] = kmeans_cluster_means(syms, filepath=filepath, packet=p)
        except Exception as e:
            print(f"    kmeans failed for packet {p}: {e}", flush=True)
            means[:, p] = 0

    if means.shape[1] > TARGET_PACKETS:
        means = means[:, :TARGET_PACKETS]
    elif means.shape[1] < TARGET_PACKETS:
        pad = TARGET_PACKETS - means.shape[1]
        means = np.pad(means, ((0, 0), (0, pad)))
    return means


def normalize_packet(data):
    for key in ['cluster_means_channel_ant1', 'cluster_means_channel_ant2']:
        arr = data[key]

        if arr is None:
            arr = np.zeros((4, TARGET_PACKETS), dtype=complex)

        arr = np.asarray(arr)

        if arr.ndim == 1:
            arr = arr[:, None]

        if arr.shape[1] > TARGET_PACKETS:
            arr = arr[:, :TARGET_PACKETS]
        elif arr.shape[1] < TARGET_PACKETS:
            pad = TARGET_PACKETS - arr.shape[1]
            arr = np.pad(arr, ((0, 0), (0, pad)))

        data[key] = arr

    data['num_packets'] = TARGET_PACKETS
    return data


def load_mat_file(filepath):

    mat = scipy.io.loadmat(filepath, struct_as_record=False, squeeze_me=True)
    if 'final_data' not in mat:
        raise ValueError(f"{filepath} missing 'final_data'")

    def inspect_mat_struct(fd):
        print("=== final_data fields ===")
        if hasattr(fd, '__dict__'):
            for k, v in fd.__dict__.items():
                if k.startswith('_'):
                    continue
                if isinstance(v, np.ndarray):
                    print(f"{k:30s} shape={v.shape} dtype={v.dtype}")
                else:
                    print(f"{k:30s} type={type(v)} value={v}")
        else:
            print("Not a structured object")

    fd = mat['final_data']

    # inspect_mat_struct(fd)

    num_packets_S1 = int(fd.num_packets_S1)
    num_packets_S2 = int(fd.num_packets_S2)
    num_packets = min(num_packets_S1, num_packets_S2)

    def ensure_2d(arr):
        arr = np.asarray(arr)
        if arr.ndim == 1:
            return arr[:, np.newaxis]
        return arr

    ant1_S1 = ensure_2d(fd.all_syms_ant1_S1)[:, :num_packets_S1]
    ant2_S1 = ensure_2d(fd.all_syms_ant2_S1)[:, :num_packets_S1]
    ant1_S2 = ensure_2d(fd.all_syms_ant1_S2)[:, :num_packets_S2]
    ant2_S2 = ensure_2d(fd.all_syms_ant2_S2)[:, :num_packets_S2]

    def trim_or_pad(arr):
        if arr.shape[1] > TARGET_PACKETS:
            return arr[:, :TARGET_PACKETS]
        elif arr.shape[1] < TARGET_PACKETS:
            pad = TARGET_PACKETS - arr.shape[1]
            return np.pad(arr, ((0, 0), (0, pad)))
        return arr

    ant1_S1 = trim_or_pad(ant1_S1)
    ant2_S1 = trim_or_pad(ant2_S1)
    ant1_S2 = trim_or_pad(ant1_S2)
    ant2_S2 = trim_or_pad(ant2_S2)

    if num_packets_S1 == 0 or num_packets_S2 == 0:
        print("num_packets_S1:", num_packets_S1)
        print("num_packets_S2:", num_packets_S2)
        raise ValueError(f"Empty packet data in file: {filepath}")

    h_ant1_S1 = compute_channel_h(ant1_S1, REF_SYM1)
    h_ant2_S1 = compute_channel_h(ant2_S1, REF_SYM1)
    h_ant1_S2 = compute_channel_h(ant1_S2, REF_SYM2)
    h_ant2_S2 = compute_channel_h(ant2_S2, REF_SYM2)

    data_S1 = {
        'num_packets': num_packets,
        'cluster_means_channel_ant1': compute_cluster_means(h_ant1_S1, filepath),
        'cluster_means_channel_ant2': compute_cluster_means(h_ant2_S1, filepath),
    }

    data_S2 = {
        'num_packets': num_packets,
        'cluster_means_channel_ant1': compute_cluster_means(h_ant1_S2, filepath),
        'cluster_means_channel_ant2': compute_cluster_means(h_ant2_S2, filepath),
    }

    return normalize_packet(data_S1), normalize_packet(data_S2)


def parse_filename(filepath):
    """
    Expected format:
     mimo_tx1_0_0_45_asdfjkdsalfj.mat
            | | | |
           tx x y angle
    """
    fname = os.path.basename(filepath)
    fname = fname.replace('n', '-')
    match = re.search(
                r'tx(\d+)_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)_(-?\d+).*\.mat',
                fname
            )
    if match:
        tx_id = int(match.group(1))
        x = float(match.group(2))
        y = float(match.group(3))
        angle = int(match.group(4))
        return tx_id, x, y, angle
    return None, None, None, None


def group_files(data_dir):
    files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith('.mat')
    ])

    groups = defaultdict(lambda: defaultdict(dict))

    count = 0
    for f in files:
        count += 1
        tx_id, x, y, angle = parse_filename(f)
        if tx_id is not None:
            groups[(x, y)][tx_id][angle] = f
    print("num files found:", count)

    i = 0
    for (x, y), tx_dict in groups.items():
        for tx_id, angle_dict in tx_dict.items():
            if len(angle_dict) != 4:
                print(f"(x,y)=({x},{y}), tx={tx_id} has {len(angle_dict)}/4 angles")
        if (len(tx_dict) != 3):
            print(f"(x,y)=({x},{y}), has less than 3 tx positions: {list(tx_dict.keys())}")
        print(f"{i}:({x},{y})")
        i += 1

    print("num (x,y) points found", len(groups))
    return groups


ANGLES = [0, 45, 90, 135]


def build_feature_matrix(groups):
    """
    Returns:
        X : (N, 4, 6, 2, 4, target_packets) complex
            dims: (sample, angle, tx/antenna, rx_antenna, cluster, packet)
        y : (N, 3)  [x, y, z]
    """
    X, y = [], []

    for (x, y_pos) in sorted(groups.keys()):
        tx_dict = groups[(x, y_pos)]

        if len(tx_dict) != 3:
            continue

        print(f"\nGoing through (x,y)=({x},{y_pos})")

        angle_features_all = []

        for angle in ANGLES:
            tx_features = []

            for tx_id in sorted(tx_dict.keys()):
                angle_dict = tx_dict[tx_id]

                if angle not in angle_dict:
                    print(f"Missing angle {angle} for tx={tx_id} at (x,y)=({x},{y_pos})")
                    break

                try:
                    data_S1, data_S2 = load_mat_file(angle_dict[angle])
                except Exception as e:
                    print(f"Skipping file {angle_dict[angle]}: \n{e}")
                    break

                for data in [data_S1, data_S2]:  # expands 3 TX → 6 TX
                    ant_data = []

                    def safe_2d(x, fallback_packets=TARGET_PACKETS):
                        if x is None:
                            return np.zeros((4, fallback_packets), dtype=complex)
                        x = np.asarray(x)
                        if x.size == 0:
                            return np.zeros((4, fallback_packets), dtype=complex)
                        if x.ndim == 1:
                            x = x[:, None]
                        if x.ndim != 2:
                            return np.zeros((4, fallback_packets), dtype=complex)
                        return x

                    if 1 in USE_ANTENNAS:
                        ant_data.append(safe_2d(data['cluster_means_channel_ant1']))
                    if 2 in USE_ANTENNAS:
                        ant_data.append(safe_2d(data['cluster_means_channel_ant2']))

                    ant_data = [
                        np.asarray(a) if a.ndim > 1 else a[:, None]
                        for a in ant_data
                    ]

                    clean_ant_data = []
                    for a in ant_data:
                        if a is None:
                            a = np.zeros((4, TARGET_PACKETS), dtype=complex)
                        else:
                            a = np.asarray(a)
                            if a.size == 0:
                                a = np.zeros((4, TARGET_PACKETS), dtype=complex)
                            if a.ndim == 1:
                                a = a[:, None]
                        clean_ant_data.append(a)

                    ant_data = clean_ant_data
                    ch = np.stack(ant_data, axis=1)  # (4 clusters, 2 rx_ants, packets)
                    tx_features.append(ch)

                    shapes = [t.shape for t in tx_features]
                    if len(set(shapes)) != 1:
                        print(f"[SKIP] TX shape mismatch: {shapes}")

            if len(tx_features) != 6:
                break

            angle_features_all.append(np.stack(tx_features, axis=0))  # (6, 4, 2, packets)

        if len(angle_features_all) != 4:
            print(f"Skipping (x,y)=({x},{y_pos}), incomplete angles")
            continue

        sample = np.stack(angle_features_all, axis=0)   # (4, 6, 4, 2, packets)
        sample = np.transpose(sample, (0, 1, 3, 2, 4))  # (4 angles, 6 tx, 2 rx_ants, 4 clusters, packets)
        X.append(sample)
        y.append([x * 0.6096, y_pos * 0.6096, 0.571 * 0.6096])

    if not X:
        return np.empty((0, 4, 6, len(USE_ANTENNAS), 4, TARGET_PACKETS), dtype=complex), np.empty((0, 3))

    return np.array(X), np.array(y)  # (N, 4, 6, 2, 4, 8), (N, 3)


# ==============================================================================
# Polar flatten  —  all r columns before all phi columns
# ==============================================================================

def flatten_and_save_polar(X, y, trainortest):
    """
    Converts complex values to polar form (r, phi) and saves as CSV.

    X shape: (N, 4 angles, 6 tx, 2 rx_ants, 4 clusters, 8 packets)

    features_polar.csv columns:
      feat_r_0   ... feat_r_{M-1}    (all magnitudes)
      feat_phi_0 ... feat_phi_{M-1}  (all phases, in same traversal order as r)
      where M = 4 * 6 * 2 * 4 * 8 = 1536

    labels_polar.csv columns: x, y, z
    """
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    N = X.shape[0]
    flat = X.reshape(N, -1)   # (N, 1536) complex

    r   = np.abs(flat)        # magnitude
    phi = np.angle(flat)      # phase in [-pi, pi]
    
    # Normalize r to [0, 1]
    r_max = np.maximum(r_train.max(), 1e-8)
    print("r_max", r_max)
    r = r / r_max
    
    phi = (phi + np.pi) / (2 * np.pi)  # [-pi, pi] → [0, 1]

    n_feat = flat.shape[1]    # 1536
    r_names   = [f'feat_r_{i}'   for i in range(n_feat)]
    phi_names = [f'feat_phi_{i}' for i in range(n_feat)]

    df_X = pd.DataFrame(
        np.hstack([r, phi]),
        columns=r_names + phi_names
    )
    df_y = pd.DataFrame(y, columns=['x', 'y', 'z'])

    xname = f'pilot2{trainortest}_features_polar_1norm_eachdata.csv'
    yname = f'pilot2{trainortest}_labels.csv'

    df_X.to_csv(os.path.join(output_dir, xname), index=False)
    df_y.to_csv(os.path.join(output_dir, yname), index=False)

    print(f"Saved X: {df_X.shape} -> {xname}")
    print(f"Saved y: {df_y.shape} -> {yname}")


# ==============================================================================
# FUTURE: relative-phi flatten
# ==============================================================================

def flatten_and_save_polar_relative(X, y, output_dir, features_file, labels_file):
    """
    Relative-phase polar features.

    X shape: (N, 4 angles, 6 tx, 2 rx_ants, 4 clusters, 8 packets)

    For each (angle, tx, cluster, packet) cell the two rx-antenna values
    (ant1, ant2) collapse into three features:
        r1, r2, phi2 - phi1

    The full traversal order (outermost → innermost) is:
        angle → tx → cluster → packet
    and within each cell: r_ant1, r_ant2, phi_ant2 - phi_ant1

    Output feature count: 4 * 6 * 4 * 8 * 3 = 2304
    Column layout (all r1 | all r2 | all delta-phi):
        feat_r1_0  ... feat_r1_767
        feat_r2_0  ... feat_r2_767
        feat_dphi_0 ... feat_dphi_767
    where index 0..767 traverses (angle, tx, cluster, packet) in C order.
    """
    os.makedirs(output_dir, exist_ok=True)

    N = X.shape[0]
    r   = np.abs(X)
    phi = np.angle(X)

    r1   = r[:, :, :, 0, :, :]
    r2   = r[:, :, :, 1, :, :]
    phi1 = phi[:, :, :, 0, :, :]
    phi2 = phi[:, :, :, 1, :, :]

    dphi = phi2 - phi1
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    dphi = (dphi + np.pi) / (2 * np.pi)

    r_max = np.maximum(r.max(axis=(1,2,3,4,5)), 1e-8)
    r_max = r_max.reshape(N, 1, 1, 1, 1)
    r1 = r1 / r_max
    r2 = r2 / r_max

    r1_flat   = r1.reshape(N, -1)
    r2_flat   = r2.reshape(N, -1)
    dphi_flat = dphi.reshape(N, -1)

    n_cells = r1_flat.shape[1]
    df_X = pd.DataFrame(
        np.hstack([r1_flat, r2_flat, dphi_flat]),
        columns=[f'feat_r1_{i}' for i in range(n_cells)]
               + [f'feat_r2_{i}' for i in range(n_cells)]
               + [f'feat_dphi_{i}' for i in range(n_cells)]
    )
    df_y = pd.DataFrame(y, columns=['x', 'y', 'z'])

    features_path = os.path.join(output_dir, features_file)
    labels_path   = os.path.join(output_dir, labels_file)
    df_X.to_csv(features_path, index=False)
    df_y.to_csv(labels_path,   index=False)

    print(f"Saved X: {df_X.shape} -> {features_path}")
    print(f"Saved y: {df_y.shape} -> {labels_path}")


def flatten_and_save_polar_1norm(X, y, X_train, output_dir, features_file, labels_file):
    os.makedirs(output_dir, exist_ok=True)

    N = X.shape[0]
    flat = X.reshape(N, -1)
    r   = np.abs(flat)
    phi = np.angle(flat)

    flat_train = X_train.reshape(X_train.shape[0], -1)
    r_max = np.maximum(np.abs(flat_train).max(), 1e-8)
    print("r_max", r_max)
    r   = r / r_max
    phi = (phi + np.pi) / (2 * np.pi)

    n_feat = flat.shape[1]
    df_X = pd.DataFrame(
        np.hstack([r, phi]),
        columns=[f'feat_r_{i}'   for i in range(n_feat)]
               + [f'feat_phi_{i}' for i in range(n_feat)]
    )
    df_y = pd.DataFrame(y, columns=['x', 'y', 'z'])

    features_path = os.path.join(output_dir, features_file)
    labels_path   = os.path.join(output_dir, labels_file)
    df_X.to_csv(features_path, index=False)
    df_y.to_csv(labels_path,   index=False)

    print(f"Saved X: {df_X.shape} -> {features_path}")
    print(f"Saved y: {df_y.shape} -> {labels_path}")


def flatten_and_save_polar_relative_1norm(X, y, X_train, output_dir, features_file, labels_file):
    os.makedirs(output_dir, exist_ok=True)

    N = X.shape[0]

    r_max = np.maximum(np.abs(X_train).max(), 1e-8)
    print("r_max", r_max)

    r   = np.abs(X)
    phi = np.angle(X)

    r1   = r[:, :, :, 0, :, :]
    r2   = r[:, :, :, 1, :, :]
    phi1 = phi[:, :, :, 0, :, :]
    phi2 = phi[:, :, :, 1, :, :]

    r1 = r1 / r_max
    r2 = r2 / r_max

    dphi = phi2 - phi1
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    dphi = (dphi + np.pi) / (2 * np.pi)

    r1_flat   = r1.reshape(N, -1)
    r2_flat   = r2.reshape(N, -1)
    dphi_flat = dphi.reshape(N, -1)

    n_cells = r1_flat.shape[1]
    df_X = pd.DataFrame(
        np.hstack([r1_flat, r2_flat, dphi_flat]),
        columns=[f'feat_r1_{i}'   for i in range(n_cells)]
               + [f'feat_r2_{i}'   for i in range(n_cells)]
               + [f'feat_dphi_{i}' for i in range(n_cells)]
    )
    df_y = pd.DataFrame(y, columns=['x', 'y', 'z'])

    features_path = os.path.join(output_dir, features_file)
    labels_path   = os.path.join(output_dir, labels_file)
    df_X.to_csv(features_path, index=False)
    df_y.to_csv(labels_path,   index=False)

    print(f"Saved X: {df_X.shape} -> {features_path}")
    print(f"Saved y: {df_y.shape} -> {labels_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir',      default='../F1_MIMO_train_processed',
                        help='Directory containing train .mat files', required = False)
    parser.add_argument('--test_dir',       default='../F1_MIMO_test_processed',
                        help='Directory containing test .mat files', required = False)
    parser.add_argument('--output_dir',     default='../MIMO_feature-csvs',
                        help='Directory to write output CSVs', required = False)
    parser.add_argument('--train_features', default='train_features.csv',
                        help='Filename for the train features CSV', required = False)
    parser.add_argument('--train_labels',   default='train_labels.csv',
                        help='Filename for the train labels CSV', required = False)
    parser.add_argument('--test_features',  default='test_features.csv',
                        help='Filename for the test features CSV', required = False)
    parser.add_argument('--test_labels',    default='test_labels.csv',
                        help='Filename for the test labels CSV', required = False)
    parser.add_argument('--mode',
                        choices=['polar_relative', 'polar_1norm', 'polar_relative_1norm'],
                        default='polar_relative',
                        help='Which flatten function to run', required = True)
    args = parser.parse_args()

    # Load train
    train_groups = group_files(args.train_dir)
    X_train, y_train = build_feature_matrix(train_groups)
    print(f"X_train shape : {X_train.shape}")
    print(f"y_train shape : {y_train.shape}")

    # Load test
    test_groups = group_files(args.test_dir)
    X_test, y_test = build_feature_matrix(test_groups)
    print(f"X_test shape : {X_test.shape}")
    print(f"y_test shape : {y_test.shape}")

    if args.mode == 'polar_relative':
        flatten_and_save_polar_relative(X_train, y_train, args.output_dir, args.train_features, args.train_labels)
        flatten_and_save_polar_relative(X_test,  y_test,  args.output_dir, args.test_features,  args.test_labels)

    elif args.mode == 'polar_1norm':
        flatten_and_save_polar_1norm(X_train, y_train, X_train, args.output_dir, args.train_features, args.train_labels)
        flatten_and_save_polar_1norm(X_test,  y_test,  X_train, args.output_dir, args.test_features,  args.test_labels)

    elif args.mode == 'polar_relative_1norm':
        flatten_and_save_polar_relative_1norm(X_train, y_train, X_train, args.output_dir, args.train_features, args.train_labels)
        flatten_and_save_polar_relative_1norm(X_test,  y_test,  X_train, args.output_dir, args.test_features,  args.test_labels)