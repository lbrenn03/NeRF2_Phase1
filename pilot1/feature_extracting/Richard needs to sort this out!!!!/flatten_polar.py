import numpy as np
import pandas as pd
import os
from flatten import group_files, build_feature_matrix


def flatten_and_save_polar(X, y):
    """
    Converts complex values to polar form (r, phi) and saves as CSV.
    
    features.csv columns: feat_r_0 ... feat_r_191, feat_phi_0 ... feat_phi_191
    labels.csv columns: x, y, z
    """

    output_dir='.'

    N = X.shape[0]
    flat = X.reshape(N, -1)

    r   = np.abs(flat)        # magnitude
    phi = np.angle(flat)      # phase arctan(b/a)

    n_feat = flat.shape[1]
    r_names   = [f'feat_r_{i}'   for i in range(n_feat)]
    phi_names = [f'feat_phi_{i}' for i in range(n_feat)]

    df_X = pd.DataFrame(
        np.hstack([r, phi]),
        columns=r_names + phi_names
    )
    df_y = pd.DataFrame(y, columns=['x', 'y', 'z'])

    df_X.to_csv(os.path.join(output_dir, 'features_polar.csv'), index=False)
    df_y.to_csv(os.path.join(output_dir, 'labels.csv'), index=False)

    print(f"Saved X: {df_X.shape} -> features_polar.csv")
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

    flatten_and_save_polar(X, y)