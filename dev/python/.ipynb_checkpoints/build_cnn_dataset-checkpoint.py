import argparse
import numpy as np
import os

# Reuse pipeline from original script
from flatten import group_files, build_feature_matrix 

def reshape_for_cnn(X: np.ndarray) -> np.ndarray:
    """
    X : (N, 3, 4, 2, 8) complex
    Returns : (N, 2, 12, 16) float32
    """
    N = X.shape[0]

    X_2d = X.reshape(N, 3 * 4, 2 * 8)

    # Step 2: split complex into two real channels
    X_cnn = np.stack([X_2d.real, X_2d.imag], axis=1)   # (N, 2, 12, 16)

    return X_cnn.astype(np.float32)


def save_cnn_dataset(X_cnn: np.ndarray, y: np.ndarray, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    x_path = os.path.join(out_dir, "X_cnn.npy")
    y_path = os.path.join(out_dir, "y_cnn.npy")

    np.save(x_path, X_cnn)
    np.save(y_path, y.astype(np.float32))

    print(f"X_cnn : {X_cnn.shape}  +  {x_path}")   # (N, 2, 12, 16)
    print(f"y_cnn : {y.shape}      +  {y_path}")    # (N, 3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Directory containing preprocessed .mat files")
    parser.add_argument("--out_dir", default=".", help="Output directory for .npy files")
    args = parser.parse_args()

    groups = group_files(args.data_dir)
    X, y = build_feature_matrix(groups)

    print(f"Raw  X : {X.shape}")
    print(f"NaN count : {np.isnan(X.real).sum()}")

    X_cnn = reshape_for_cnn(X)
    save_cnn_dataset(X_cnn, y, args.out_dir)