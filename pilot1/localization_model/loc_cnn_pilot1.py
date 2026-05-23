"""
================================================================================
loc_cnn_pilot1.py — RF-Based Indoor Localization via Convolutional Neural Network
================================================================================

Overview
--------
This script trains and evaluates a 2D CNN (LocalizationCNN) that predicts
3-D receiver positions (x, y, z) in meters from radio-frequency (RF) channel
measurements collected in a pilot indoor-localization experiment ("pilot1").

Pipeline
--------
1. Load a feature matrix (CSV) and a label matrix (CSV) for each dataset
   variant (RSSI, raw channel coefficients, polar features, synthetic, etc.).
2. Reshape the flat feature rows into 2-D spatial tensors that the CNN can
   convolve over (axes: TX position × antenna × OFDM symbol × packet index).
3. Train LocalizationCNN with AdamW + Cosine-Annealing-Warm-Restarts LR
   scheduling, using the test split as a validation set.
4. Evaluate the best model: Euclidean distance error, CDF, scatter plots.
5. Save predictions, plots, and a summary CSV of mean errors per dataset.

Key Tensor Shape Convention (after reshaping)
---------------------------------------------
  (N, C, H, W)  where
    N  = number of samples
    C  = number of input channels fed to Conv2d
         2 → real + imaginary parts of complex measurements
         1 → magnitude-only measurements
         3 → three-channel polar features (r1, r2, Δφ)
    H  = N_TX * N_ANT   (spatial "height" of the 2-D feature map)
    W  = N_SYM * N_PKT  (temporal/packet "width" of the 2-D feature map)

Dataset Dimension Tuple: (N_COMPLEX, N_TX, N_SYM, N_ANT, N_PKT)
  N_COMPLEX : number of feature channels / parts (real+imag = 2, mag only = 1,
              polar 3-channel = 3)
  N_TX      : number of transmitter positions in the room (always 3)
  N_SYM     : number of OFDM pilot symbols per packet
  N_ANT     : number of receive antennas (1 or 2)
  N_PKT     : number of packets / snapshots per measurement

Dependencies
------------
  PyTorch, NumPy, pandas, matplotlib
  Custom plotting helpers: loc_plotting.py

  
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

import os
import pandas as pd
import tqdm
import random
import numpy as np
import copy

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

from loc_plotting import *   # dataset-specific scatter / CDF / histogram helpers

# ---------------------------------------------------------------------------
# Top-level directory layout
# ---------------------------------------------------------------------------
datadir  = "../feature_datasets"    # root directory for all raw CSV feature/label files
logsdir  = "logs"    # output directory for checkpoints, prediction CSVs, etc.
plotsdir = "plots"   # output directory for all generated figures

# ---------------------------------------------------------------------------
# Dataset dimension tuples  (N_COMPLEX, N_TX, N_SYM, N_ANT, N_PKT)
#
# Each tuple describes the logical shape of one sample's feature vector
# BEFORE it is reshaped into the (C, H, W) format expected by the CNN.
# ---------------------------------------------------------------------------
just_rssi_dims = (1, 3, 1, 2, 1)  # scalar RSSI only: 1 channel, 3 TX, 1 sym, 2 ant, 1 pkt
just_r_dims    = (1, 3, 4, 2, 8)  # magnitude |r| only: 1 channel, 3 TX, 4 sym, 2 ant, 8 pkts
all_dims       = (2, 3, 4, 2, 8)  # complex a+jb: 2 channels (real, imag), full resolution
phi_rel_dims   = (3, 3, 4, 1, 8)  # polar relative: 3 channels (r1, r2, Δφ), 1 antenna
mixed_dims     = (2, 3, 1, 2, 8)  # mixed real+imag but only 1 OFDM symbol
syn_dims       = (2, 3, 1, 2, 1)  # synthetic data: complex, 1 sym, 1 pkt

# ---------------------------------------------------------------------------
# Dataset paths — RSSI-only baseline (Luke-generated, pilot 1)
# ---------------------------------------------------------------------------
group = "pilot1_rssi"
rssi_dir_test       = os.path.join(datadir, group, f'rssi.csv')           # averaged RSSI feature matrix
labels_dir_test     = os.path.join(datadir, group, f'rx_pos_meters.csv')  # 3-D ground-truth positions
train_idx_dir_test  = os.path.join(datadir, group, f'train_index.txt')    # pre-saved train split indices
test_idx_dir_test   = os.path.join(datadir, group, f'test_index.txt')     # pre-saved test split indices

# Shared label / index files used by most raw-channel datasets
labels_dir    = os.path.join(datadir, f'pilot1_labels.csv')          # ground-truth (x,y,z) for raw datasets
train_idx_dir = os.path.join(datadir, f'pilot1_train_index.txt')     # train split indices (shared)
est_idx_dir   = os.path.join(datadir, f'pilot1_test_index.txt')      # test split indices  (shared)

# ---------------------------------------------------------------------------
# |r| (magnitude) features only — 4 symbols
# ---------------------------------------------------------------------------
group = "pilot1_r"
r_dir = os.path.join(datadir, group, f'features_justr.csv')          # per-sample magnitude values

# ---------------------------------------------------------------------------
# Polar features: r (magnitude) + φ (phase) as complex number, normalised
# ---------------------------------------------------------------------------
group        = "pilot1_r_phi"
rphi_dir     = os.path.join(datadir, group, f'features_polar_norm.csv')    # normalised (r, φ) polar form
rphi_rel_dir = os.path.join(datadir, group, f'features_r1_r2_dphi.csv')    # differential: (r1, r2, Δφ)

# ---------------------------------------------------------------------------
# Complex channel coefficients a + jb
# ---------------------------------------------------------------------------
group  = "pilot1_a_b"
ab_dir = os.path.join(datadir, group, f'features.csv')               # real part a | imag part b

# ---------------------------------------------------------------------------
# Channel-division and channel-multiplication features
# (dividing / multiplying channel matrices across TX pairs to capture
#  relative phase/amplitude differences between transmitters)
# ---------------------------------------------------------------------------
group          = "pilot1_channel_div"
channeldiv_dir = os.path.join(datadir, group, f'features_channel_div.csv')  # TX-pair ratio features
group          = "pilot1_channel_mul"
channelmul_dir = os.path.join(datadir, group, f'features_channel_mul.csv')  # TX-pair product features

# ---------------------------------------------------------------------------
# Synthetic a+jb data (Tien-generated ray-tracing / simulation)
# ---------------------------------------------------------------------------
group           = "pilot1_tien_syn_ab"
syn_ab_dir      = os.path.join(datadir, group, f'a_b_synthetic_features_data.csv')  # purely synthetic samples
syn_labels_dir  = os.path.join(datadir, group, f'a_b_synthetic_labels.csv')         # synthetic ground-truth
syn_train_idx_dir = os.path.join(datadir, group, f'pilot1_syn_train_index.txt')
syn_test_idx_dir  = os.path.join(datadir, group, f'pilot1_syn_test_index.txt')

# Mixed = synthetic + real measurements blended together
tienmix_ab_dir       = os.path.join(datadir, group, f'a_b_mixed_features_data.csv')
tienmix_labels_dir   = os.path.join(datadir, group, f'a_b_mixed_labels.csv')
tienmix_train_idx_dir = os.path.join(datadir, group, f'pilot1_mixed_train_index.txt')
tienmix_test_idx_dir  = os.path.join(datadir, group, f'pilot1_mixed_test_index.txt')

# ---------------------------------------------------------------------------
# Mixed real a+jb in 2-D and 3-D (different feature aggregations)
# ---------------------------------------------------------------------------
group              = "pilot1_mixed_ab"
mixed_labels_dir   = os.path.join(datadir, group, f'labels_abi_3d_mixed.csv')
mixed_train_idx_dir = os.path.join(datadir, group, f'pilot1_mixed_train_index.txt')
mixed_test_idx_dir  = os.path.join(datadir, group, f'pilot1_mixed_test_index.txt')
mixed_ab3_dir      = os.path.join(datadir, group, f'features_abi_3d_mixed.csv')   # 3-D (includes z-component)
mixed_ab2_dir      = os.path.join(datadir, group, f'features_abi_2d_mixed.csv')   # 2-D (xy plane only)
mixed_ch2_dir      = os.path.join(datadir, group, f'features_channel_div_mixed_2d.csv')  # channel-div 2-D mixed

# ---------------------------------------------------------------------------
# Dataset registry tuples
# Format: (dataset_name, feature_csv, label_csv, train_idx, test_idx, dim_tuple)
#
# dataset_name : unique string key; also controls model architecture selection
# feature_csv  : path to flat feature matrix  (N × F)
# label_csv    : path to ground-truth positions (N × 3)
# train_idx    : path to .txt file with training sample indices
# test_idx     : path to .txt file with test sample indices
# dim_tuple    : (N_COMPLEX, N_TX, N_SYM, N_ANT, N_PKT) for reshaping
# ---------------------------------------------------------------------------

# RSSI-only baseline
raw_dirs    = ("pilot1_rssi",          rssi_dir_test, labels_dir_test, train_idx_dir_test, test_idx_dir_test, just_rssi_dims)

# Magnitude-only: full resolution, single-symbol (one), averaged-symbol, 2-D (sym+pkt averaged)
r_dirs      = ("pilot1_r",             r_dir, labels_dir, train_idx_dir, est_idx_dir, just_r_dims)
r3one_dirs  = ("pilot1_r_3dim_one",    r_dir, labels_dir, train_idx_dir, est_idx_dir, just_r_dims)  # single sym idx
r3avg_dirs  = ("pilot1_r_3dim_avg",    r_dir, labels_dir, train_idx_dir, est_idx_dir, just_r_dims)  # sym averaged
r2_dirs     = ("pilot1_r_2dim",        r_dir, labels_dir, train_idx_dir, est_idx_dir, just_r_dims)  # sym+pkt averaged

# Polar (r, φ) — full, 3-D, 2-D
rphi_dirs   = ("pilot1_r_phi",         rphi_dir, labels_dir, train_idx_dir, est_idx_dir, all_dims)
rphi3_dirs  = ("pilot1_r_phi_3dim",    rphi_dir, labels_dir, train_idx_dir, est_idx_dir, all_dims)
rphi2_dirs  = ("pilot1_r_phi_2dim",    rphi_dir, labels_dir, train_idx_dir, est_idx_dir, all_dims)

# Polar relative Δφ — full, 3-D, 2-D
rphi_rel_dirs  = ("pilot1_r_phi_rel",       rphi_rel_dir, labels_dir, train_idx_dir, est_idx_dir, phi_rel_dims)
rphi3_rel_dirs = ("pilot1_r_phi_rel_3dim",  rphi_rel_dir, labels_dir, train_idx_dir, est_idx_dir, phi_rel_dims)
rphi2_rel_dirs = ("pilot1_r_phi_rel_2dim",  rphi_rel_dir, labels_dir, train_idx_dir, est_idx_dir, phi_rel_dims)

# Complex a+jb — full, 3-D, 2-D
ab_dirs  = ("pilot1_a_b",       ab_dir, labels_dir, train_idx_dir, est_idx_dir, all_dims)
ab3_dirs = ("pilot1_a_b_3dim",  ab_dir, labels_dir, train_idx_dir, est_idx_dir, all_dims)
ab2_dirs = ("pilot1_a_b_2dim",  ab_dir, labels_dir, train_idx_dir, est_idx_dir, all_dims)

# Channel division — full, 3-D, 2-D
channeldiv_dirs  = ("pilot1_channeldiv",       channeldiv_dir, labels_dir, train_idx_dir, est_idx_dir, all_dims)
channeldiv3_dirs = ("pilot1_channeldiv_3dim",  channeldiv_dir, labels_dir, train_idx_dir, est_idx_dir, all_dims)
channeldiv2_dirs = ("pilot1_channeldiv_2dim",  channeldiv_dir, labels_dir, train_idx_dir, est_idx_dir, all_dims)

# Channel multiplication — full, 3-D, 2-D
channelmul_dirs  = ("pilot1_channelmul",       channelmul_dir, labels_dir, train_idx_dir, est_idx_dir, all_dims)
channelmul3_dirs = ("pilot1_channelmul_3dim",  channelmul_dir, labels_dir, train_idx_dir, est_idx_dir, all_dims)
channelmul2_dirs = ("pilot1_channelmul_2dim",  channelmul_dir, labels_dir, train_idx_dir, est_idx_dir, all_dims)

# Synthetic and mixed datasets
syn_ab_dirs      = ("pilot1_tien_syn_ab_2dim",     syn_ab_dir,     syn_labels_dir,   syn_train_idx_dir,    syn_test_idx_dir,    syn_dims)
tienmix_ab_dirs  = ("pilot1_tien_mix_ab_2dim",     tienmix_ab_dir, tienmix_labels_dir, tienmix_train_idx_dir, tienmix_test_idx_dir, syn_dims)
mixed_ab3_dirs   = ("pilot1_mixed_ab_3dim",        mixed_ab3_dir,  mixed_labels_dir, mixed_train_idx_dir,  mixed_test_idx_dir,  mixed_dims)
mixed_ab2_dirs   = ("pilot1_mixed_ab_2dim",        mixed_ab2_dir,  mixed_labels_dir, mixed_train_idx_dir,  mixed_test_idx_dir,  syn_dims)
mixed_ch2_dirs   = ("pilot1_mixed_channel_2dim",   mixed_ch2_dir,  mixed_labels_dir, mixed_train_idx_dir,  mixed_test_idx_dir,  syn_dims)

# ---------------------------------------------------------------------------
# Master list of dataset configs to iterate over during training.
# Comment out entries to skip specific datasets.
# ---------------------------------------------------------------------------
dirs = [raw_dirs, r_dirs, r3avg_dirs, r3one_dirs, r2_dirs,
        rphi_dirs, rphi3_dirs, rphi2_dirs,
        rphi_rel_dirs, rphi3_rel_dirs, rphi2_rel_dirs,
        ab_dirs, ab3_dirs, ab2_dirs,
        channeldiv_dirs, channeldiv3_dirs, channeldiv2_dirs,
        channelmul_dirs, channelmul3_dirs, channelmul2_dirs,
        syn_ab_dirs, tienmix_ab_dirs, mixed_ab3_dirs, mixed_ab2_dirs, mixed_ch2_dirs]

# ---------------------------------------------------------------------------
# Global training hyper-parameters
# ---------------------------------------------------------------------------
num_epoch = 200   # total training epochs per dataset / seed
lr        = 0.001 # initial learning rate for AdamW optimizer

import random
import numpy as np
import torch

def set_seed(seed=42):
    """
    Fix all sources of randomness to make experiments reproducible.

    Sets seeds for Python's `random`, NumPy, and PyTorch (both CPU and all
    CUDA devices).  Also disables cuDNN's non-deterministic auto-tuner.

    Args:
        seed (int): Integer seed value. Default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True   # reproducible conv results
    torch.backends.cudnn.benchmark     = False  # disables auto-tuner (trades speed for reproducibility)

set_seed(42)

class LocalizationCNN(nn.Module):
    """
    2-D Convolutional Neural Network for RF-based 3-D indoor localization.

    Architecture overview
    ---------------------
    Input  : (B, C, H, W) feature map
               B = batch size
               C = input_channels (1, 2, or 3 depending on feature type)
               H = N_TX * N_ANT   (spatial height)
               W = N_SYM * N_PKT  (temporal width)

    Layers:
      Conv block 1 : Conv2d → BN → ReLU  ×2  → MaxPool (pool1)
      Conv block 2 : Conv2d → BN → ReLU  ×2  → MaxPool (pool2)
                     (skipped via nn.Identity for datasets where ignore=True)
      AdaptiveAvgPool2d → flatten → FC1 → Dropout → FC2 → Dropout → FC_out

    Output : (B, 3) — predicted (x, y, z) position in metres

    Dataset-specific branching
    --------------------------
    `dataset_name` selects:
      - `model_param`    : [model_sz, model_sz2, fc1_sz, fc2_sz]
                           channel widths for conv layers and FC sizes
      - `pool1_k_sz`     : kernel size for the first MaxPool
      - `pool2_k_sz`     : kernel size (tuple) for the second MaxPool
      - `ignore`         : whether to replace conv block 2 with identity layers
                           (used for low-dimensional inputs that would spatially
                            collapse before reaching block 2)

    Args:
        input_channels (int): Number of input feature channels C. Default 2.
        dataset_name   (str): Key string used to select architecture variant.
    """
    def __init__(self, input_channels=2, dataset_name=""):
        super().__init__()
        output_dim = 3  # predict (x, y, z) coordinates

        # ------------------------------------------------------------------
        # Architecture variant selection based on dataset_name
        # ------------------------------------------------------------------
        ignore = False  # when True, conv block 2 becomes identity (no-op)

        # model_param = [conv_channels_block1, conv_channels_block2, fc1_size, fc2_size]
        if dataset_name == "pilot1_rssi":
            model_param = [128, 128, 512, 512]
            ignore = True  # RSSI has only 1 spatial cell; skip block 2
        elif dataset_name == "pilot1_r":
            model_param = [128, 256, 256, 128]
        elif dataset_name in ("pilot1_r_3dim_one", "pilot1_r_3dim_avg"):
            model_param = [128, 128, 256, 128]
            ignore = True  # single-symbol magnitude collapses spatial dim
        elif dataset_name == "pilot1_r_phi":
            model_param = [256, 512, 512, 512]
        elif dataset_name == "pilot1_r_phi_3dim":
            model_param = [256, 256, 1024, 512]
        elif dataset_name == "pilot1_r_phi_2dim":
            model_param = [256, 256, 256, 256]
        elif dataset_name == "pilot1_r_phi_rel":
            model_param = [128, 256, 1024, 512]
        elif dataset_name == "pilot1_r_phi_rel_3dim":
            model_param = [64, 64, 256, 256]
            ignore = True
        elif dataset_name == "pilot1_r_phi_rel_2dim":
            model_param = [256, 256, 256, 128]
        elif dataset_name == "pilot1_a_b":
            model_param = [256, 256, 1024, 512]
        elif dataset_name == "pilot1_a_b_2dim":
            model_param = [128, 256, 512, 256]
        elif dataset_name in ("pilot1_channel", "pilot1_channeldiv", "pilot1_channelmul"):
            model_param = [256, 256, 512, 512]
        elif dataset_name in ("pilot1_channel_3dim", "pilot1_channeldiv_3dim"):
            model_param = [64, 64, 256, 128]
        elif dataset_name == "pilot1_channelmul_3dim":
            model_param = [128, 128, 256, 256]
        elif dataset_name == "pilot1_channel_2dim":
            model_param = [128, 256, 256, 128]
        elif dataset_name.find("2dim") != -1:
            # Catch-all for any remaining "2dim" variants
            model_param = [128, 128, 256, 128]
            ignore = True  # 2-D variants have collapsed pkt dimension
        else:
            model_param = [256, 256, 1024, 512]  # default (full-resolution)

        # Unpack layer widths
        model_sz, model_sz2, fc1_sz, fc2_sz = model_param

        # ------------------------------------------------------------------
        # Pooling kernel sizes — chosen to match spatial resolution of each
        # dataset variant and avoid over-pooling small feature maps
        # ------------------------------------------------------------------
        if dataset_name.find("2dim") != -1 or dataset_name == "pilot1_rssi":
            # 2-D / RSSI inputs have a collapsed packet dimension (W=1);
            # pooling with stride>1 along that axis would remove all info
            pool1_k_sz = 1        # no downsampling after block 1
            pool2_k_sz = (1, 1)   # no downsampling after block 2
        elif dataset_name.find("rel") != -1:
            # Relative-phase datasets have a narrower temporal extent
            pool1_k_sz = 2        # halve height only
            pool2_k_sz = (1, 2)   # halve width only
        else:
            # Full-resolution datasets: standard 2×2 downsampling
            pool1_k_sz = 2
            pool2_k_sz = (2, 2)

        conv_k_sz = 3  # 3×3 convolution kernel (with padding=1 → same spatial size)

        # ------------------------------------------------------------------
        # Layer definitions
        # ------------------------------------------------------------------
        # --- Convolutional block 1 ---
        self.conv1 = nn.Conv2d(input_channels, model_sz,  kernel_size=conv_k_sz, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(model_sz)
        self.conv2 = nn.Conv2d(model_sz,       model_sz,  kernel_size=conv_k_sz, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(model_sz)
        self.pool1 = nn.MaxPool2d(kernel_size=pool1_k_sz, stride=pool1_k_sz)

        # --- Convolutional block 2 (may be replaced by identity below) ---
        self.conv3 = nn.Conv2d(model_sz,  model_sz2, kernel_size=conv_k_sz, stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(model_sz2)
        self.conv4 = nn.Conv2d(model_sz2, model_sz2, kernel_size=conv_k_sz, stride=1, padding=1)
        self.bn4   = nn.BatchNorm2d(model_sz2)
        self.pool2 = nn.MaxPool2d(kernel_size=pool2_k_sz, stride=(1, 1))  # stride=1 → shape-preserving

        # --- Global pooling: collapse spatial dims to 1×1 regardless of input size ---
        self.adapt = nn.AdaptiveAvgPool2d((1, 1))

        # --- Fully-connected head ---
        self.fc1 = nn.Linear(model_sz2, fc1_sz)  # first hidden FC layer
        self.fc2 = nn.Linear(fc1_sz,    fc2_sz)  # second hidden FC layer
        self.out = nn.Linear(fc2_sz,    output_dim)  # regression output: (x, y, z)

        # --- Utility layers ---
        self.flatten = nn.Flatten()          # flatten (B, C, 1, 1) → (B, C)
        self.dropout = nn.Dropout(p=0.2)     # 20 % dropout for regularisation
        self.relu    = nn.ReLU()

        # Replace conv block 2 with identity when the spatial map is already
        # too small to benefit from a second conv stage (avoids NaN from
        # zero-sized tensors after pooling)
        if ignore:
            self.conv3 = nn.Identity()
            self.bn3   = nn.Identity()
            self.conv4 = nn.Identity()
            self.bn4   = nn.Identity()

    def forward(self, x):
        """
        Forward pass through the CNN.

        Args:
            x (Tensor): shape (B, C, H, W) — e.g. (B, 2, 6, 32) for the
                        full complex a+b dataset with 3 TX × 2 ant and
                        4 sym × 8 pkt.

        Returns:
            Tensor: shape (B, 3) — predicted (x, y, z) in metres.
        """
        # Convolutional block 1: extract low-level spatial features
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)  # spatial downsampling (or no-op for 2-D/RSSI)

        # Convolutional block 2: deeper feature extraction
        # (no-op / identity for datasets flagged with ignore=True)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)  # further (optional) downsampling

        # Collapse variable spatial size → fixed (1, 1) before FC layers
        x = self.adapt(x)

        # Fully-connected regression head
        x = self.flatten(x)          # (B, model_sz2)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.out(x)           # (B, 3)


class RSSIDataset():
    """
    Minimal PyTorch-compatible dataset wrapper for pre-loaded NumPy arrays.

    Args:
        inputs (np.ndarray): Feature array of shape (N, C, H, W).
        labels (np.ndarray): Label array  of shape (N, 3) — (x, y, z) metres.
    """
    def __init__(self, inputs, labels):
        self.inputs = torch.from_numpy(inputs).float()  # float32 tensor
        self.labels = torch.from_numpy(labels).float()  # float32 tensor (raw metres)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


def split_dataset(datadir, rssi_dir, train_index_dir, test_index_dir, ratio=0.8):
    """
    Randomly shuffle all samples and write 80/20 train/test index files.

    Only called when the index files do not already exist on disk.

    Args:
        datadir         (str): Root data directory (unused but kept for context).
        rssi_dir        (str): Path to the feature CSV (used only to get N).
        train_index_dir (str): Output path for the train index .txt file.
        test_index_dir  (str): Output path for the test  index .txt file.
        ratio         (float): Fraction of samples assigned to training. Default 0.8.
    """
    data  = pd.read_csv(rssi_dir)
    index = np.arange(len(data))
    random.shuffle(index)

    train_len   = int(len(index) * ratio)
    train_index = np.array(index[:train_len])   # first 80 % of shuffled indices
    test_index  = np.array(index[train_len:])   # remaining 20 %

    np.savetxt(train_index_dir, train_index, fmt='%s')
    np.savetxt(test_index_dir,  test_index,  fmt='%s')


def train_model(train_loader, val_loader=None, epochs=10, lr=0.0001,
                dataset_name="", skip=True, input_channels=2, seed=42):
    """
    Train LocalizationCNN and track loss history.

    Optimiser  : AdamW with weight decay 1e-2 (L2 regularisation).
    Scheduler  : CosineAnnealingWarmRestarts — restarts LR every T_0=20 epochs,
                 decaying to eta_min=1e-6 between restarts.
    Loss       : MSE between predicted and true (x, y, z) positions.
    Early stop : Counter-based patience=20 epochs without val improvement
                 (currently DISABLED — the break is commented out so training
                  always runs for the full `epochs` count).
    Best model : State dict of the epoch with lowest validation loss is kept
                 in `best_model_state` (NOTE: not returned — the model returned
                 is the last-epoch state, not the best).

    Args:
        train_loader   (DataLoader): Training set batches.
        val_loader     (DataLoader): Validation/test set batches (optional).
        epochs         (int)       : Total number of training epochs.
        lr             (float)     : Initial learning rate.
        dataset_name   (str)       : Architecture selector string.
        skip           (bool)      : Unused parameter (reserved for future use).
        input_channels (int)       : Number of input feature channels C.
        seed           (int)       : RNG seed for full reproducibility.

    Returns:
        model (LocalizationCNN): Trained model (last epoch weights).
        hist  (dict): Loss history with keys:
            "train_loss"        — per-epoch mean training loss (MSE)
            "val_loss"          — per-epoch mean validation loss (MSE)
            "train_batch_losses"— list-of-lists of per-batch training losses
            "val_batch_losses"  — list-of-lists of per-batch validation losses
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model     = LocalizationCNN(input_channels=input_channels, dataset_name=dataset_name).to(device)
    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    # T_0=20: restart period in epochs; T_mult=1: period stays constant after each restart
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1, eta_min=1e-6)

    hist = {
        "train_loss":         [],  # mean training MSE per epoch
        "val_loss":           [],  # mean validation MSE per epoch
        "train_batch_losses": [],  # raw per-batch losses for each epoch
        "val_batch_losses":   [],  # raw per-batch val losses for each epoch
    }

    safe_lr   = f"{lr:.0e}"               # e.g. "1e-03" — used in checkpoint filenames
    ckpt_dir  = f"{logsdir}/ckpts"
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val   = float("inf")  # lowest validation loss seen so far
    patience   = 20            # epochs to wait before early stopping (unused)
    counter    = 0             # epochs without improvement since best_val
    best_epoch = 0             # epoch index at which best_val was achieved

    for epoch in range(1, epochs + 1):
        epoch_train_batch_losses = []   # per-batch losses for this epoch
        model.train()
        running_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)   # MSE between predicted and true positions
            loss.backward()
            optimizer.step()
            epoch_train_batch_losses.append(loss.item())
            running_loss += loss.item() * x.size(0)   # accumulate weighted loss

        # Step scheduler once per epoch (not per batch) — Tien's recommendation
        scheduler.step()

        epoch_loss = running_loss / len(train_loader.dataset)  # epoch mean MSE
        hist["train_loss"].append(epoch_loss)
        hist["train_batch_losses"].append(epoch_train_batch_losses.copy())

        # ------------------------------------------------------------------
        # Validation loop
        # ------------------------------------------------------------------
        if val_loader is not None:
            epoch_val_batch_losses = []
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    batch_loss = criterion(model(x), y).item()
                    epoch_val_batch_losses.append(batch_loss)
                    val_loss += batch_loss * x.size(0)

            val_loss /= len(val_loader.dataset)   # epoch mean validation MSE
            hist["val_loss"].append(val_loss)
            hist["val_batch_losses"].append(epoch_val_batch_losses)

            # Track best model state (lowest validation loss)
            if val_loss < best_val:
                best_epoch      = epoch
                best_val        = val_loss
                counter         = 0
                best_model_state = copy.deepcopy(model.state_dict())  # deep copy so later updates don't overwrite
            else:
                counter += 1

            # Early stopping check (currently disabled — break is commented out)
            # if counter >= patience:
            #     print(f"Early stopping at epoch {epoch}")
            #     break

            if epoch % 10 == 0:
                print(f"Epoch {epoch:03d}: train={epoch_loss:.6f}, val={val_loss:.6f}")
        else:
            if epoch % 10 == 0:
                print(f"Epoch {epoch:03d}: train={epoch_loss:.6f}")
            hist["val_loss"].append(None)         # placeholder when no val set
            hist["val_batch_losses"].append(None)

    return model, hist


def test_model(model, test_loader, device=None, save_csv=False, csv_path="predictions.csv"):
    """
    Evaluate a trained model on a held-out test set and report Euclidean errors.

    Metrics computed
    ----------------
    * Per-sample Euclidean distance error: ‖ pred − true ‖₂  (in metres)
    * Mean MSE across all samples
    * Mean Euclidean error
    * Median Euclidean error
    * Count and details of samples with error > `threshold` metres

    Args:
        model       (LocalizationCNN): Trained model to evaluate.
        test_loader (DataLoader)     : DataLoader for the test set.
        device      (torch.device)   : Inference device (default: GPU if available).
        save_csv    (bool)           : If True, write predictions + errors to CSV.
        csv_path    (str)            : Output CSV path (used only if save_csv=True).

    Returns:
        predictions (np.ndarray): Shape (N, 3) — predicted (x, y, z).
        targets     (np.ndarray): Shape (N, 3) — ground-truth (x, y, z).
        errors      (np.ndarray): Shape (N,)   — per-sample Euclidean error.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    predictions = []  # accumulate batch predictions
    targets     = []  # accumulate batch ground-truth labels

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            predictions.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())

    predictions = np.vstack(predictions)  # (N, 3)
    targets     = np.vstack(targets)      # (N, 3)

    # Per-sample squared error averaged over (x, y, z) components
    mse_per_sample = np.mean((predictions - targets) ** 2, axis=1)
    mean_mse       = np.mean(mse_per_sample)

    # Euclidean (L2) distance in 3-D space
    errors       = np.linalg.norm(predictions - targets, axis=1)  # (N,)
    mean_error   = np.mean(errors)
    median_error = np.median(errors)

    # Flag predictions that are far off — may indicate outliers or failure cases
    threshold      = 30  # metres; samples beyond this distance are printed
    high_error_idx = np.where(errors > threshold)[0]

    print(f"\nNumber of samples with error > {threshold}: {len(high_error_idx)}")

    for idx in high_error_idx:
        print(f"\nSample {idx}")
        print(f"  True: {targets[idx]}")
        print(f"  Pred: {predictions[idx]}")
        print(f"  Error: {errors[idx]:.2f}")

    if save_csv:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df = pd.DataFrame({
            "x_pred": predictions[:, 0],
            "y_pred": predictions[:, 1],
            "z_pred": predictions[:, 2],
            "x_true": targets[:, 0],
            "y_true": targets[:, 1],
            "z_true": targets[:, 2],
            "error":  errors
        })
        df.to_csv(csv_path, index=False)
        print(f"Saved predictions and errors to {csv_path}")

    print(f"Mean Euclidean error: {mean_error:.4f}")
    print(f"Mean MSE: {mean_mse:.6f}")
    print(f"Median Error: {median_error:.6f}")
    return predictions, targets, errors


# ===========================================================================
# Main training and evaluation loop
# ===========================================================================
if __name__ == "__main__":
    set_seed(42)

    # Dictionary of model variants to train.
    # Key   : descriptive label used in logs and plots.
    # Value : bool — currently unused (residual variant is disabled).
    model_variants = {
        "baseline": False
        # "residual": True  # placeholder for a future residual-block variant
    }

    mean_errors = []   # list of dicts; one entry per (dataset, seed) run
    # CDF error accumulator keyed by variant name → {run_key: errors array}
    cdf_errors = {"baseline": {}}

    for dataset_name, inputs_dir, labels_dir, train_index_dir, test_index_dir, dims in dirs:
        # ------------------------------------------------------------------
        # Load or generate train/test split indices
        # ------------------------------------------------------------------
        if not os.path.exists(train_index_dir) or not os.path.exists(test_index_dir):
            split_dataset(datadir, inputs_dir, train_index_dir, test_index_dir, ratio=0.8)

        train_index = np.loadtxt(train_index_dir, dtype=int)  # 1-D array of sample indices
        test_index  = np.loadtxt(test_index_dir,  dtype=int)

        inputs = pd.read_csv(inputs_dir)   # flat feature matrix  (N × F_flat)
        labels = pd.read_csv(labels_dir)   # ground-truth positions (N × 3)

        # ------------------------------------------------------------------
        # Feature reshaping into CNN-compatible (N, C, H, W) format
        #
        # The flat CSV layout differs by N_COMPLEX:
        #   N_COMPLEX == 2: first half of columns → real part,
        #                   second half             → imaginary part
        #   N_COMPLEX == 3: all columns are already separated into 3 channels
        #                   (r1, r2, Δφ), stored consecutively
        #   N_COMPLEX == 1: raw real-valued magnitudes, no split needed
        # ------------------------------------------------------------------
        N_COMPLEX, N_TX, N_SYM, N_ANT, N_PKT = dims

        N = inputs.shape[0]   # total number of measurement samples
        print("INPUT shape: ", inputs.shape)

        if N_COMPLEX == 2:
            # Complex a+jb stored as [real_cols | imag_cols] in the CSV
            N_FEAT = N_TX * N_SYM * N_ANT * N_PKT  # number of features per channel
            X = inputs.iloc[:, :N_FEAT].values + 1j * inputs.iloc[:, N_FEAT:].values  # (N, N_FEAT) complex

        elif N_COMPLEX == 3:
            # Three-channel polar features (r1, r2, Δφ) laid out flat
            N_FEAT = N_COMPLEX * N_TX * N_SYM * N_PKT   # total feature columns
            X = inputs.values
            X = X.reshape(N, N_COMPLEX, N_TX, N_SYM, N_PKT)  # (N, 3, 3, 4, 8) for full-res

            # Dataset-specific dimension reductions for 3-channel variants
            if dataset_name == "pilot1_r_phi_rel_3dim":
                sym_idx = 0  # use only the first OFDM symbol
                X = X[:, :, :, sym_idx:sym_idx+1, :]  # (N, 3, 3, 1, 8)
                N_SYM = 1

            elif dataset_name == "pilot1_r_phi_rel_2dim":
                print("shrink dimension")
                sym_idx = 0
                X = X[:, :, :, sym_idx:sym_idx+1, :]  # (N, 3, 3, 1, 8) — single symbol
                N_SYM = 1
                X = X.mean(axis=4, keepdims=True)      # average over packets → (N, 3, 3, 1, 1)
                N_PKT = 1

            # Flatten the last two spatial axes into a single width axis
            X = X.reshape(N, N_COMPLEX, N_TX, N_SYM * N_PKT)  # (N, 3, 3, W)
            inputs_cnn = X  # shape is already (N, C, H, W) for N_COMPLEX==3

        else:
            # N_COMPLEX == 1: magnitude-only or RSSI, no real/imag split
            N_FEAT = N_TX * N_SYM * N_ANT * N_PKT
            X = inputs.values  # (N, N_FEAT) real-valued

        # ------------------------------------------------------------------
        # For N_COMPLEX != 3, perform dataset-specific axis slicing /
        # averaging before stacking into the final CNN input tensor
        # ------------------------------------------------------------------
        if N_COMPLEX != 3:
            # Interpret flat columns as (TX, SYM, ANT, PKT) axes
            X = X.reshape(N, N_TX, N_SYM, N_ANT, N_PKT)   # (N, 3, 4, 2, 8) for full-res

            # Dataset-specific symbol / packet averaging or slicing
            if dataset_name in ("pilot1_a_b_3dim", "pilot1_r_phi_3dim", "pilot1_r_3dim_one"):
                # Keep only the first symbol to reduce temporal extent
                sym_idx = 0
                X = X[:, :, sym_idx:sym_idx+1, :, :]  # (N, 3, 1, 2, 8)
                N_SYM = 1

            elif dataset_name == "pilot1_r_3dim_avg":
                # Average over all symbols → single pseudo-symbol
                X = X.mean(axis=2, keepdims=True)      # (N, 3, 1, 2, 8)
                N_SYM = 1

            elif dataset_name == "pilot1_r_2dim":
                # Average over both symbols and packets → single scalar per (TX, ANT)
                X = X.mean(axis=2, keepdims=True)      # sym avg → (N, 3, 1, 2, 8)
                N_SYM = 1
                X = X.mean(axis=4, keepdims=True)      # pkt avg → (N, 3, 1, 2, 1)
                N_PKT = 1

            elif dataset_name in ("pilot1_channeldiv_3dim", "pilot1_channelmul_3dim"):
                sym_idx = 0
                X = X[:, :, sym_idx:sym_idx+1, :, :]
                N_SYM = 1

            elif dataset_name in ("pilot1_channeldiv_2dim", "pilot1_channelmul_2dim"):
                # Single symbol, then average over packets
                sym_idx = 0
                X = X[:, :, sym_idx:sym_idx+1, :, :]
                N_SYM = 1
                X = X.mean(axis=4, keepdims=True)
                N_PKT = 1

            elif dataset_name in ("pilot1_a_b_2dim", "pilot1_r_phi_2dim"):
                sym_idx = 0
                X = X[:, :, sym_idx:sym_idx+1, :, :]
                N_SYM = 1
                X = X.mean(axis=4, keepdims=True)
                N_PKT = 1

            # Reorder axes so antennas interleave with TX positions in the
            # height dimension: (N, TX, ANT, SYM, PKT) → merge TX×ANT and SYM×PKT
            X = X.transpose(0, 1, 3, 2, 4)           # (N, N_TX, N_ANT, N_SYM, N_PKT)
            X = X.reshape(N, N_TX * N_ANT, N_SYM * N_PKT)  # (N, H, W) where H=6, W≤32

            if N_COMPLEX == 2:
                # Stack real and imaginary parts as separate channels
                inputs_cnn = np.stack([X.real, X.imag], axis=1)  # (N, 2, H, W)
            else:
                # Single channel: add a channel dim
                inputs_cnn = X[:, None, :, :]                     # (N, 1, H, W)

        # ------------------------------------------------------------------
        # Build DataLoaders from the pre-split index arrays
        # ------------------------------------------------------------------
        test_inputs  = inputs_cnn[test_index]
        test_labels  = labels.iloc[test_index].reset_index(drop=True).values
        test_dataset = RSSIDataset(test_inputs, test_labels)
        test_loader  = DataLoader(test_dataset, batch_size=128, shuffle=False)

        train_inputs  = inputs_cnn[train_index]
        train_labels  = labels.iloc[train_index].reset_index(drop=True).values
        train_dataset = RSSIDataset(train_inputs, train_labels)

        # ------------------------------------------------------------------
        # Training loop over random seeds (currently a single seed: 42)
        # ------------------------------------------------------------------
        for seed in range(42, 42 + 1):
            g = torch.Generator()
            g.manual_seed(seed)  # seed the DataLoader shuffle for reproducibility
            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, generator=g)

            history_list = []   # accumulates hist dicts across seeds (for multi-seed plots)
            label_list   = []   # corresponding legend labels
            results_dir  = f"{plotsdir}"

            print(f"\n==== Seed {seed}: Training on {dataset_name} dataset with "
                  f"LR = {lr} on {num_epoch} epochs ====\n")

            model, hist = train_model(
                train_loader, val_loader=test_loader,
                epochs=num_epoch, lr=lr,
                dataset_name=dataset_name,
                skip=False,
                input_channels=N_COMPLEX,
                seed=seed
            )

            # Save per-sample predictions to CSV for offline analysis
            csv_name = f"{logsdir}/predictions/pred_{dataset_name}_{seed}.csv"
            predictions, targets, errors = test_model(
                model, test_loader,
                save_csv=True,
                csv_path=csv_name
            )

            # Store errors for combined CDF plot at the end
            cdf_errors["baseline"][f"{dataset_name}_{seed}"] = errors

            # Scatter plot: predicted vs true positions with connecting lines
            xy_save_path = (f"{results_dir}/{dataset_name}/seed_{seed}/"
                            f"diff_{dataset_name}_{seed}.png")
            plot_xy_prediction_lines(
                predictions, targets,
                train_xy=train_labels,
                title=f"Prediction difference w/ target - {dataset_name} data",
                save_path=xy_save_path
            )

            # Error histogram for quick visual inspection of error distribution
            plot_error_histogram(errors, save_path=f"{results_dir}/{dataset_name}/testerr.png")

            # Record scalar summary for the final results table
            mean_error = float(np.mean(errors))
            mean_errors.append({
                "dataset":              dataset_name,
                "dimensions":           f"{N} x {N_COMPLEX} x {N_TX} x {N_ANT} x {N_SYM} x {N_PKT}",
                "seed":                 seed,
                "mean_euclidean_error": mean_error
            })

            history_list.append(hist)
            label_list.append(f"lr={lr}_{num_epoch}ep")

            # Training / validation loss curves
            plot_results(history_list, label_list, dataset_name,
                         save_dir=f"{results_dir}/{dataset_name}")
            # Epoch-level variance across seeds (single seed → no variance here)
            plot_epoch_variability(history_list, label_list, dataset_name,
                                   save_dir=f"{results_dir}/{dataset_name}")

    # ------------------------------------------------------------------
    # After all datasets: combined CDF overlay across all dataset variants
    # ------------------------------------------------------------------
    plot_multiple_cdfs(cdf_errors["baseline"], save_path=f"{plotsdir}/cnn_cdf_overlay.png")

    # ------------------------------------------------------------------
    # Print and save the final performance summary table
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(mean_errors)
    results_df["mean_euclidean_error"] = results_df["mean_euclidean_error"].map(lambda x: f"{x:.3f}")

    print(f"\n===== MODEL PERFORMANCE TABLE (Mean Errors) with LR = {lr} on {num_epoch} epochs =====")
    print(results_df.to_string(
        index=False,
        justify='left',
        formatters={
            "dataset":   lambda x: f"{x:<24}",
            "dimensions": lambda x: f"{x:<25}",
            "mean_error": lambda x: f"{x:<6.3f}",
        }
    ))

    results_df.to_csv(f"{logsdir}/cnn_mean_errors.csv", index=False)
    print(f"\nSaved table to {logsdir}/cnn_mean_errors.csv")