import numpy as np
import pandas as pd
import os
from flatten import group_files, load_mat_file

def flatten_and_save(X, y): 
    output_dir= '.'

    N = X.shape[0]
    flat = X.reshape(N, -1)  # (N, 3*4*2*8) complex

    n_feat = flat.shape[1]
    real_names = [f'feat_real_{i}' for i in range(n_feat)]
    imag_names = [f'feat_imag_{i}' for i in range(n_feat)]

    df_X = pd.DataFrame(
        np.hstack([flat.real, flat.imag]),
        columns=real_names + imag_names
    )
    df_y = pd.DataFrame(y, columns=['x', 'y', 'z'])

    df_X.to_csv(os.path.join(output_dir, 'features_split_ant.csv'), index=False)
    df_y.to_csv(os.path.join(output_dir, 'labels_split_ant.csv'), index=False)

    print(f"Saved X: {df_X.shape} -> features.csv")
    print(f"Saved y: {df_y.shape} -> labels.csv")

def flatten_and_save(X, y):
    output_dir = '.'
    N = X.shape[0]
    flat = X.reshape(N, -1)

    n_feat = flat.shape[1]
    real_names = [f'feat_real_{i}' for i in range(n_feat)]
    imag_names = [f'feat_imag_{i}' for i in range(n_feat)]

    df_X = pd.DataFrame(
        np.hstack([flat.real, flat.imag]),
        columns=real_names + imag_names
    )
    df_y = pd.DataFrame(y, columns=['x', 'y', 'z'])

    df_X.to_csv(os.path.join(output_dir, 'features_split_ant.csv'), index=False)
    df_y.to_csv(os.path.join(output_dir, 'labels_split_ant.csv'), index=False)

    print(f"Saved X: {df_X.shape} -> features_split_ant.csv")
    print(f"Saved y: {df_y.shape} -> labels_split_ant.csv")

def build_feature_matrix(groups):
    X, y = [], []

    for rx_id in sorted(groups.keys()):
        tx_files = groups[rx_id]

        if len(tx_files) != 3:
            print(f"Skipping rx_id={rx_id}, missing TX files")
            continue

        tx_ant1 = []
        tx_ant2 = []
        rx_pos = None

        for tx_id in sorted(tx_files.keys()):
            data = load_mat_file(tx_files[tx_id])
            if data is None:
                break

            tx_ant1.append(data['cluster_means_ant1'][:, np.newaxis, :])
            tx_ant2.append(data['cluster_means_ant2'][:, np.newaxis, :])
            rx_pos = data['rx_pos'] * 0.6096

        if len(tx_ant1) != 3:
            print(f"Skipping rx_id={rx_id}, failed to load all TX files")
            continue

        X.append(np.stack(tx_ant1, axis=0))
        y.append(rx_pos - np.array([0.082, 0, 0])) # Ant 1

        X.append(np.stack(tx_ant2, axis=0))
        y.append(rx_pos + np.array([0.082, 0, 0])) # Ant 2

    return np.array(X), np.array(y)  # (222, 3, 4, 1, 8), (222, 3)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    args = parser.parse_args()

    groups = group_files(args.data_dir)
    X, y = build_feature_matrix(groups)

    print(f"NaN count in X: {np.isnan(X.real).sum()}")
    print(f"X shape : {X.shape}")
    print(f"y shape : {y.shape}")

    flatten_and_save(X, y)