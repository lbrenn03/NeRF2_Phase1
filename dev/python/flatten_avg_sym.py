import numpy as np
import pandas as pd
import re
import os
from collections import defaultdict
from scipy.cluster.vq import kmeans2

import scipy.io
import warnings

TARGET_PACKETS = 8
USE_ANTENNAS = [1, 2]


def kmeans_cluster_means(syms, filepath='unknown', packet=0, ant=0):
    pts = np.column_stack([syms.real, syms.imag])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        centroids, labels = kmeans2(pts, 4, iter=20, minit='points', missing='warn')
        # Debuggins Statement
        # if w:
        #     means = centroids[:, 0] + 1j * centroids[:, 1]
        #     print(f"  kmeans warning ant{ant} packet {packet}: {filepath}  — {w[0].message}")
        #     print(f"    centroids: {means}")
    return centroids[:, 0] + 1j * centroids[:, 1]


def normalize_packet(data):
    num_packets = int(data['num_packets'].flatten()[0])

    for key in ['all_syms_ant1', 'all_syms_ant2', 'cluster_means_ant1', 'cluster_means_ant2']:
        array = data[key]
        
        if num_packets > TARGET_PACKETS:
            data[key] = array[:, :TARGET_PACKETS] # Trim to TARGET_PACKETS
        elif num_packets < TARGET_PACKETS:
            padding = TARGET_PACKETS - num_packets

            data[key] = np.pad(
                array,
                ((0, 0), (padding, 0)),
                constant_values=np.nan
            )
    
    data['num_packets'] = TARGET_PACKETS
    return data

def ensure_cluster_means(data, filepath='unknown'):
    # if 'cluster_means_ant1' not in data or 'cluster_means_ant2' not in data:
    #     # print(f"  cluster_means missing: {filepath}")
    num_packets = data['all_syms_ant1'].shape[1]
    
    cluster_means_ant1 = np.full((4, num_packets), np.nan, dtype=complex)
    cluster_means_ant2 = np.full((4, num_packets), np.nan, dtype=complex)
    
    for p in range(num_packets):
        try:
            cluster_means_ant1[:, p] = kmeans_cluster_means(data['all_syms_ant1'][:, p], filepath=filepath, packet=p, ant=1)
        except Exception as e:
            print(f"    kmeans failed for packet {p}: {e}", flush=True)
        try:
            cluster_means_ant2[:, p] = kmeans_cluster_means(data['all_syms_ant2'][:, p], filepath=filepath, packet=p, ant=2)
        except Exception as e:
            print(f"    kmeans failed for packet {p}: {e}", flush=True)
            
    data['cluster_means_ant1'] = cluster_means_ant1
    data['cluster_means_ant2'] = cluster_means_ant2

    return data

# Extract data from processed MATLAB file
def load_mat_file(filepath):
    try:
        mat_data = scipy.io.loadmat(filepath)
        
        final_data = mat_data['final_data'][0, 0]

        data = {
            'num_packets': final_data['num_packets'], # (1, num_packets)
            'all_syms_ant1': final_data['all_syms_ant1'], # (500, num_packets)
            'all_syms_ant2': final_data['all_syms_ant2'], # (500, num_packets)
            'rx_pos': final_data['rx_pos'].flatten(),  # Convert (1, 3) to (3,)
            'tx_pos': final_data['tx_pos'].flatten(),  # Convert (1, 3) to (3,)
        }
        if 'cluster_means_ant1' in final_data.dtype.names:
            data['cluster_means_ant1'] = final_data['cluster_means_ant1'] # (4, num_packets)
        if 'cluster_means_ant2' in final_data.dtype.names:
            data['cluster_means_ant2'] = final_data['cluster_means_ant2'] # (4, num_packets)

        ensure_cluster_means(data, filepath=filepath)

        return normalize_packet(data)
    
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def parse_filename(filepath):
    match = re.search(r'tx(\d+)_(n?\d+_\d+)\.mat$', os.path.basename(filepath))
    if match:
        tx_id = int(match.group(1))   # 1, 2, or 3
        rx_id = match.group(2)        # '2_0', 'n1_3' etc.
        return tx_id, rx_id
    return None, None

def group_files(data_dir):
    files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.mat')])
    
    groups = defaultdict(dict)

    for f in files:
        tx_id, rx_id = parse_filename(f)
        if tx_id is not None:
            groups[rx_id][tx_id] = f
    
    # Warning if any group is incomplete
    for rx_id, tx_files in groups.items():
        if len(tx_files) != 3:
            print(f"rx_id={rx_id} has {len(tx_files)}/3 TX files: {list(tx_files.keys())}")
    
    return groups

def build_feature_matrix(groups):
    """
    Returns:
        X : (N rxpos, 3 txpos, 2 ant) complex
        y : (N rxpos, 3 txpos) float  [rx_pos]
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
                ant_data.append(data['cluster_means_ant1'])
            if 2 in USE_ANTENNAS:
                ant_data.append(data['cluster_means_ant2'])
            cm = np.stack(ant_data, axis=1)

            tx_features.append(cm)
            rx_pos = data['rx_pos']

        if len(tx_features) != 3:
            print(f"Skipping rx_id={rx_id}, failed to load all TX files")
            continue

        X.append(np.stack(tx_features, axis=0))  # (3, 2)
        y.append(rx_pos * 0.6096) # Unit conversion from ft to m
        
    X = np.nanmean(np.array(X), axis=4) # average packets
    X = np.nanmean(X, axis=2) # average symbols
    X = np.nan_to_num(X, nan=0.0)

    return np.array(X), np.array(y)  # (N, 3, 2), (N, 3)


def flatten_and_save(X, y): 
    """
    Splits complex values into real/imag, flattens to 2D and saves as X.csv.
    Saves rx positions separately as y.csv.

    X.csv columns: feat_real_0 ... feat_real_N, feat_imag_0 ... feat_imag_N
    y.csv columns: x, y, z
    """
    output_dir= '.'

    N = X.shape[0]
    flat = X.reshape(N, -1)  # (N, 3*2) complex

    n_feat = flat.shape[1]
    real_names = [f'feat_real_{i}' for i in range(n_feat)]
    imag_names = [f'feat_imag_{i}' for i in range(n_feat)]

    df_X = pd.DataFrame(
        np.hstack([flat.real, flat.imag]),
        columns=real_names + imag_names
    )
    df_y = pd.DataFrame(y, columns=['x', 'y', 'z'])

    df_X.to_csv(os.path.join(output_dir, 'features_sym_avg.csv'), index=False)
    df_y.to_csv(os.path.join(output_dir, 'labels.csv'), index=False)

    print(f"Saved X: {df_X.shape} -> features_sym_avg.csv")
    print(f"Saved y: {df_y.shape} -> labels.csv")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='Directory containing preprocessed .mat files')
    args = parser.parse_args()

    groups = group_files(args.data_dir)
    X, y = build_feature_matrix(groups)

    print(f"NaN count in X: {np.isnan(X.real).sum()}")

    print(f"X shape : {X.shape}")   # (N, 3, 2)
    print(f"y shape : {y.shape}")   # (N, 3)

    flatten_and_save(X, y)