import numpy as np
import pandas as pd
import os
from flatten import group_files, build_feature_matrix

def flatten_and_save_rssi(X, y):
    """
    Computes RSSI from cluster means and saves as CSV.
    RSSI = 10 * log10(mean(|cluster_means|²)) per (tx, antenna, packet)

    features_rssi.csv columns: feat_rssi_0 ... feat_rssi_47  (3*2*8=48)
    labels.csv columns: x, y, z
    """
    output_dir='.'

    N = X.shape[0]
    # X shape: (N, 3, 4, 2, 8)
    # mean over cluster dim (axis=2) → (N, 3, 2, 8)
    power = np.abs(np.mean(X, axis=2))**2 # average the 4 centroids into one complex number, then compute power
    rssi  = 10 * np.log10(power + 1e-10 )  # (N, 3, 2, 8)

    flat = rssi.reshape(N, -1)             # (N, 48)

    n_feat = flat.shape[1]
    col_names = [f'feat_rssi_{i}' for i in range(n_feat)]

    df_X = pd.DataFrame(flat, columns=col_names)
    df_y = pd.DataFrame(y, columns=['x', 'y', 'z'])

    df_X.to_csv(os.path.join(output_dir, 'features_rssi.csv'), index=False)
    df_y.to_csv(os.path.join(output_dir, 'labels.csv'), index=False)

    print(f"Saved X: {df_X.shape} -> features_rssi.csv")
    print(f"Saved y: {df_y.shape} -> labels.csv")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='Directory containing preprocessed .mat files')
    args = parser.parse_args()

    groups = group_files(args.data_dir)
    X, y = build_feature_matrix(groups)

    print(f"X shape : {X.shape}")
    print(f"y shape : {y.shape}")

    flatten_and_save_rssi(X, y)