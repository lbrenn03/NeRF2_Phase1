"""
pilot2_flatten_channel_mul.py
==============================
Loads MIMO .mat files with dual-source (S1/S2) transmissions, estimates
per-packet channel responses for each source via cross-correlation against
separate PN reference sequences, clusters with k-means, then flattens the
complex feature matrix into 2D CSVs for downstream ML pipelines.

Pipeline
--------
  .mat files  →  dual-source channel estimation (S1/S2, per antenna, per TX)
              →  k-means clustering (4 centroids per packet, per source)
              →  feature matrix  X : (N, 4, 6, 2, 4, target_packets) complex
                                     (angle, TX×source, antenna, cluster, packet)
                                 y : (N, 3) float  [x, y, z] in meters
              →  flatten + split real/imag
              →  features CSV  +  labels CSV

Key Differences from pilot1
-----------------------------------------------------
  - Two PN reference sequences (pn_bits1, pn_bits2) for S1 and S2
  - Files are grouped by (x, y) position × TX × angle (4 angles: 0°, 45°, 90°, 135°)
  - Each .mat file yields two data structs (data_S1, data_S2), expanding
    3 TX positions → 6 effective TX features per angle
  - Output shape is (N, 4, 6, 2, 4, target_packets) vs pilot1's (N, 3, 4, 2, target_packets)
  - Positions are stored in meters (tiles × 0.6096)

Output CSVs
-----------
  <features_file>  —  columns: feat_real_0 … feat_real_N, feat_imag_0 … feat_imag_N
  <labels_file>    —  columns: x, y, z

Usage
-----
  python pilot2_flatten_channel_mul.py --data_dir ../../../Pilot2_MIMO/F1_MIMO_train_processed
  
  python pilot2_flatten_channel_mul.py --data_dir ../../../Pilot2_MIMO/F1_MIMO_train_processed --features_file f1test_features_ch_mul.csv --labels_file labels.csv

Arguments
---------
  --data_dir       Directory containing preprocessed .mat files
                   (default: /cluster/tufts/nerf2robotics/shared/MIMO_test_processed)
  --output_dir     Directory to write output CSVs; created if it doesn't exist
                   (default: output)
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
#[1,1,1,1,1,1
pn_bits1 = [1,1,1,1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,1,1,1,1,1,1,0,1,1,1,0,0,0,1,0,0,1,1,1,1,1,0,0,0,1,1,0,0,1,1,1,1,1,0,1,0,1,1,0,0,1,0,1,1,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,1,1,0,0,1,0,0,1,1,0,1,0,0,0,0,1,0,0,1,0,1,0,1,0,0,0,0,1,1,1,1,0,1,0,1,1,1,0,1,0,1,1,0,1,1,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,1,1,0,0,1,1,0,0,0,0,1,0,1,0,1,1,0,1,0,1,1,1,0,0,0,1,1,0,1,1,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,0,0,1,1,1,1,0,1,1,0,1,1,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,1,0,1,1,0,1,0,1,0,1,0,0,0,1,1,1,1,1,0,1,1,1,1,0,0,1,0,0,1,0,1,1,0,0,0,0,0,1,0,0,1,1,0,0,1,0,0,0,1,0,1,0,0,0,1,1,0,1,1,0,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,1,1,1,0,1,1,1,1,1,1,1,0,0,1,0,0,0,0,1,1,0,0,0,1,0,1,1,0,1,1,1,0,1,0,0,0,0,1,1,0,1,0,1,0,1,1,0,0,1,1,1,1,0,0,1,0,1,1,0,1,1,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,1,1,0,0,0,0,0,0,1,0,1,1,0,0,0,1,0,1,0,0,1,1,1,0,1,1,0,0,1,1,1,0,0,0,1,0,1,1,1,1,1,1,0,1,0,1,0,0,0,1,0,1,1,1,0,1,1,0,1,0,1,1,0,0,0,0,1,1,0,0,1,1,0,1,1,0,1,0,1,0,0,0,0,0,1,1,1,0,1,0,0,1,1,1,1,0,1,0,0,1,1,0,1,0,1,0,0,1,0,0,1,1,1,0,0,0,0,0,1,1,1,1,1,0,0,1,1,1,0,0,1,1,0,1,1,1,1,0,1,0,0,0,1,0,1,0,1,0,1,1,0,1,1,1,1,1,0,0,0,0,1,0,0,1,1,1,0,1,0,0,0,1,1,1,0,1,0,1,1,1,1,1,0,1,1,0,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,1,0,0,1,0,1,0,1,1,0,0,0,1,1,1,0,0,1,1,1,1,1,1,1,0,1,1,0,0,0,0,1,0,0,0,1,1,0,1,0,0,1,1,1,0,0,1,0,0,1,1,1,1,0,0,0,0,1,1,0,1,1,1,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,0,1,1,1,1,1,0,1,0,0,1,0,0,1,0,1,0,0,0,0,0,0,1,1,0,1,0,0,0,1,1,0,0,1,0,1,1,1,0,1,0,0,1,0,1,1,0,1,0,0,0,1,0,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1,0,1,0,0,1,0,0,0,1,1,0,0,0,0,1,1,1,0,1,1,0,1,1,1,1,0,0,0,0,0,1,0,1,1,1,0,0,1,0,1,0,1,1,1,0,0,1,1,1,0,1,1,1,0,1,1,1,0,0,1,1,0,0,1,1,1,0,1,0,1,0,1,1,1,0,1,1,1,1,0,1,1,0,0,1,0,1,0,0,0,1,0,0,1,1,0,1,1,0,0,0,1,0,0,0,0,1,1,1,0,0,1,0,1,1,1,1,1,0,0,1,0,1,0,0,1,1,0,0,1,1,0,0,1,0,1,0,1,0,1,0,0,1,1,1,1,1,1,0,0,1,1,0,0,0,1,1,0,1,0,1,1,1,1,0,0,1,1,0,1,0,1,1,0,1,0,0,1,1,0,0,0,1,0,0,1,0,1,1,1,0,0,0,0,1,0,1,1,1,1,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,0,1,0,0,0,0,0,1,0,1,0,1,0,0,1,0,1,1,1,1,0,0,0,1,0,1,0,1,1,1,1,0,1,1,1,0,1,0,1,0,0,1,1,0,1,1,1,0,0,1,0,0,0,1,1,1,0,0,0,1]
#[0,1,1,0,0,1
pn_bits2 = [1,0,1,0,1,0,1,0,0,1,0,1,0,0,1,0,0,1,0,1,1,0,1,1,1,0,1,0,1,0,1,0,0,0,0,1,1,0,1,0,1,1,1,0,1,0,0,1,1,0,0,0,1,1,1,0,1,0,0,1,1,1,0,0,0,1,1,0,0,0,1,0,1,1,1,0,0,1,0,0,0,1,1,0,0,1,1,0,0,0,1,0,1,1,1,0,1,1,0,1,0,1,0,0,0,1,0,0,0,0,1,0,1,1,1,0,0,0,0,0,1,1,1,0,1,1,0,1,0,0,0,0,1,1,0,0,1,0,0,1,1,1,0,0,0,0,1,0,1,0,1,0,0,1,0,1,0,1,1,0,0,1,0,0,0,0,1,0,1,0,1,0,1,0,0,0,0,1,0,1,1,1,1,1,0,0,1,0,1,0,0,0,0,0,0,1,1,0,1,0,1,0,1,0,0,0,0,1,0,0,1,1,1,1,0,0,1,0,1,1,1,0,0,0,1,1,1,0,0,1,0,0,0,1,1,0,1,1,1,1,0,0,0,0,1,1,0,0,1,1,0,1,0,1,0,0,1,0,0,1,1,0,0,0,0,1,1,1,0,0,1,0,1,1,0,0,0,1,0,0,0,0,1,0,0,0,1,1,0,0,1,1,0,1,0,1,1,1,0,1,0,0,1,0,0,1,0,1,0,1,1,0,1,0,0,1,0,0,0,1,0,0,0,1,1,0,0,0,0,1,1,1,0,1,1,0,1,1,0,1,1,1,0,1,0,0,1,0,0,0,0,1,0,1,1,1,0,0,0,0,0,0,0,1,0,0,1,1,1,1,0,0,0,0,1,0,1,0,1,0,1,1,0,0,0,1,0,0,0,1,1,0,0,0,1,0,1,0,1,0,0,1,0,1,1,1,1,1,0,1,1,0,0,0,0,0,1,0,1,0,1,0,0,0,1,0,0,0,0,1,0,1,1,0,1,1,1,0,1,0,0,0,1,1,0,0,1,1,1,0,0,0,1,1,1,0,1,0,0,1,0,1,0,1,0,1,0,0,0,0,0,0,1,1,1,0,0,1,0,0,1,0,1,1,1,1,0,0,1,0,1,1,1,1,0,0,1,1,1,1,1,1,0,1,0,1,1,0,1,1,0,1,0,1,0,0,1,0,0,1,0,0,0,1,0,1,0,1,0,0,0,0,1,1,0,0,0,1,0,0,1,1,1,1,1,0,0,1,1,0,1,0,0,1,1,1,1,1,1,1,1,0,1,1,0,1,1,1,1,0,0,1,0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,0,0,1,1,1,0,1,1,0,0,0,0,0,1,1,1,0,1,0,0,1,0,0,0,1,0,1,0,1,1,0,0,1,1,1,1,0,0,1,1,0,1,1,1,1,1,0,0,1,0,1,1,1,1,1,0,1,0,1,1,0,1,0,0,1,0,1,0,1,0,0,0,0,0,0,0,1,0,1,1,1,0,1,0,0,1,0,0,1,1,0,0,0,1,1,1,0,0,0,0,0,1,1,1,0,0,1,0,0,0,0,1,0,1,1,1,0,1,0,0,1,1,1,0,0,1,1,1,1,0,0,0,0,1,1,0,0,0,1,0,0,0,1,1,0,1,0,1,1,1,0,0,0,0,1,1,1,1,1,1,1,0,1,1,0,1,1,0,1,0,0,0,0,0,0,0,1,1,0,1,0,0,1,1,1,0,0,1,0,1,0,0,0,1,1,0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,1,0,1,0,1,0,0,1,1,0,1,1,1,0,1,1,1,1,1,1,0,1,1,1,1,1,0,0,0,1,0,0,0,1,1,1,0,1,0,1,1,1,0,1,1,0,1,0,0,0,0,0,1,0,1,1,0,1,1,1,1,1,0,0,0,1,0,1,0,1,0,1,0,0,0,1,1,0,0,1,1,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,1,1,1,0,0,0,0,1,0,0,0,0,0,0,1,1,1,1,0,0,0,1,0,1,0,1,0,0,0,0,1,1,1,1,0,1,1,1,1,0,0,0,0,1,0,0,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,1,0,0,0,1,1,0,1,1,1,0,0,0,0,0,1,0,1,1,0,1,1,0,1,1,1,0,0,0,0,1,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,1,0,1,0,0,0,0,1,0,0,1,1,0,1]

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

# assert False


def compute_channel_h(all_syms, REF_SYMS):
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
        #if best_start != 0:
            #print(f"Warning: best_start={best_start} (expected 0)")
        #if best_start:
            #print(best_start)
        #assert best_start == 0, "We want this data match the flatten_channel_mul.py"

        if best_start + n_syms > len(REF_SYMS):
            raise ValueError(
                f"Alignment overflow: start={best_start}, n_syms={n_syms}, ref_len={len(REF_SYMS)}"
            )
        
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
        syms = channel_h[:, p]
    
        # Skip degenerate packets
        if syms is None or len(syms) == 0:
            #print(f"Skipping bad packet {p} in {filepath}")
            means[:, p] = 0
            continue
        
        syms = np.asarray(syms)
        
        if syms.size < 10 or np.allclose(syms, 0):
            #print(f"Skipping bad packet {p} in {filepath}")
            means[:, p] = 0
            continue
        
        if np.isnan(syms).any():
            #print(f"Skipping bad packet {p} in {filepath}")
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

    # Extract symbols
    def ensure_2d(arr):
        arr = np.asarray(arr)
        if arr.ndim == 1:
            return arr[:, np.newaxis]  # (N,) → (N,1)
        return arr

    ant1_S1 = ensure_2d(fd.all_syms_ant1_S1)[:, :num_packets_S1]
    ant2_S1 = ensure_2d(fd.all_syms_ant2_S1)[:, :num_packets_S1]
    ant1_S2 = ensure_2d(fd.all_syms_ant1_S2)[:, :num_packets_S2]
    ant2_S2 = ensure_2d(fd.all_syms_ant2_S2)[:, :num_packets_S2]

    TARGET_PACKETS = 8

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
    
    # Compute channels
    h_ant1_S1 = compute_channel_h(ant1_S1, REF_SYM1)
    h_ant2_S1 = compute_channel_h(ant2_S1, REF_SYM1)
    h_ant1_S2 = compute_channel_h(ant1_S2, REF_SYM2)
    h_ant2_S2 = compute_channel_h(ant2_S2, REF_SYM2)

    # Cluster separately
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

    # Debug check
    i = 0
    for (x, y), tx_dict in groups.items():
        for tx_id, angle_dict in tx_dict.items():
            if len(angle_dict) != 4:
                print(f"(x,y)=({x},{y}), tx={tx_id} has {len(angle_dict)}/4 angles")
        if (len(tx_dict) != 3):
            print(f"(x,y)=({x},{y}), has less than 3 tx positions: {list(tx_dict.keys())}")
        print(f"{i}:({x},{y})")
        i+=1

    print("num (x,y) points found", len(groups))
    return groups


ANGLES = [0, 45, 90, 135]

def build_feature_matrix(groups):
    """
    Returns:
        X : (N, 4, 6, 2, 4, target_packets)
            (angle, tx, antenna, cluster, packet)
        y : (N, 2)  [x, y]
    """
    X, y = [], []

    for (x, y_pos) in sorted(groups.keys()):
        tx_dict = groups[(x, y_pos)]

        if len(tx_dict) != 3:
            print(len(tx_dict))
            #print(f"Skipping (x,y)=({x},{y_pos}), missing TX files")
            continue

        print(f"\nGoing through (x,y)=({x},{y_pos})")
        
        # Collect per-angle data first
        angle_features_all = []

        for angle in ANGLES:
            tx_features = []

            for tx_id in sorted(tx_dict.keys()):
                angle_dict = tx_dict[tx_id]
                # print(f"  tx={tx_id}, angle={angle}, file={angle_dict.get(angle, 'MISSING')}")
            
                if angle not in angle_dict:
                    print(f"Missing angle {angle} for tx={tx_id} at (x,y)=({x},{y_pos})")
                    continue
            
                try:
                    data_S1, data_S2 = load_mat_file(angle_dict[angle])
                except Exception as e:
                    print(f"[EMPTY/BAD FILE] {angle_dict[angle]}: {e}")
                    continue

                for data in [data_S1, data_S2]:  # ← expands 3 TX → 6 TX
                    ant_data = []

                    def safe_2d(x, fallback_packets=TARGET_PACKETS):
                        if x is None:
                            return np.zeros((4, fallback_packets), dtype=complex)
                    
                        x = np.asarray(x)
                    
                        if x.size == 0:
                            return np.zeros((4, fallback_packets), dtype=complex)
                    
                        if x.ndim == 1:
                            # ambiguous case: treat as single packet
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
                    # (4 clusters, 2 antennas, packets)
                    ch = np.stack(ant_data, axis=1)

                    tx_features.append(ch)
                    shapes = [t.shape for t in tx_features]
                    if len(set(shapes)) != 1:
                        print(tx_features)
                        print(f"[SKIP] TX shape mismatch: {shapes}")

            if len(tx_features) != 6:
                continue

            # (3 TX, 4 clusters, 2 antennas, packets)
            angle_features_all.append(np.stack(tx_features, axis=0))
    

        if len(angle_features_all) == 4:
            sample = np.stack(angle_features_all, axis=0)
            # current: (4, 3, 4, 2, packets)
            sample = np.transpose(sample, (0, 1, 3, 2, 4))
            # new:     (4 angles, 3 TX, 2 antennas, 4 clusters, packets)
            X.append(sample)
            y.append([(x+1) * 0.6096, y_pos * 0.6096, 0.571 * 0.6096])
            # y.append([x, y_pos, 0.571]) in tiles
        else:
            print(f"Skipping (x,y)=({x},{y_pos}), incomplete angles")

    if not X:
        return np.empty((0, 4, 3, 4, len(USE_ANTENNAS), TARGET_PACKETS), dtype=complex), np.empty((0, 2))

    return np.array(X), np.array(y)

def flatten_and_save(X, y, output_dir, features_file, labels_file):
    """
    Splits complex values into real/imag, flattens to 2D and saves as CSV.
    features_channel.csv columns: feat_real_0 ... feat_real_N, feat_imag_0 ... feat_imag_N
    labels_channel.csv columns: x, y, z
    """
    N = X.shape[0]
    flat = X.reshape(N, -1)  # (N, 4*6*2*4*target_packets) complex
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
                        default='../F1_MIMO_train_processed', required=True)
    parser.add_argument('--output_dir', help='Directory to save output CSV files',
                        default='../feature_datasets', required=False)
    parser.add_argument('--features_file', help='Filename for the features CSV',
                        default='features_ch_mul.csv', required=False)
    parser.add_argument('--labels_file', help='Filename for the labels CSV',
                        default='labels.csv', required=False)
    args = parser.parse_args()

    groups = group_files(args.data_dir)
    X, y = build_feature_matrix(groups)
    print(f"X shape : {X.shape}")   # (N, 4, 6, 2, 4, 8)
    print(y)
    print(f"y shape : {y.shape}")   # (N, 3)
    flatten_and_save(X, y, args.output_dir, args.features_file, args.labels_file)