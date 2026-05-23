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

from loc_plotting import *

datadir = "data"
logsdir = "logs"
plotsdir = "plots"

# DIMENSIONS for what the dimension of the !!!!!!!!!!*****INPUT FILE******!!!!!!!! is
just_rssi_dims = (1, 3, 1, 2, 1) # real/complex, tx pos, sym, ant, pkt
just_r_dims = (1, 3, 4, 2, 8)
all_dims = (2, 3, 4, 2, 8)
phi_rel_dims = (3, 3, 4, 1, 8)
mixed_dims = (2, 3, 1, 2, 8)
syn_dims = (2, 3, 1, 2, 1)

# DATASET DIRECTORY PATH (group is folder name)
# first test, averaged rssi, luke generated
group = "pilot1_rssi"
rssi_dir_test = os.path.join(datadir, group, f'rssi.csv')
labels_dir_test = os.path.join(datadir, group, f'rx_pos_meters.csv')
train_idx_dir_test = os.path.join(datadir, group, f'train_index.txt')
test_idx_dir_test = os.path.join(datadir, group, f'test_index.txt')

# same labels for raw data
labels_dir = os.path.join(datadir, f'pilot1_labels.csv')
train_idx_dir = os.path.join(datadir, f'pilot1_train_index.txt')
test_idx_dir = os.path.join(datadir, f'pilot1_test_index.txt')

# average rssi only 4 symbols
group = "pilot1_r"
r_dir = os.path.join(datadir, group, f'features_justr.csv')

# rssi and phi complex number
group = "pilot1_r_phi"
rphi_dir = os.path.join(datadir, group, f'features_polar_norm.csv')
rphi_rel_dir = os.path.join(datadir, group, f'features_r1_r2_dphi.csv')

# a and b complex number
group = "pilot1_a_b"
ab_dir = os.path.join(datadir, group, f'features.csv')

#channel data
group = "pilot1_channel_mul"
channelmul_dir = os.path.join(datadir, group, f'features_channel_mul.csv')

#pilot 2
pilot2_oneorientation_dims = (2, 6, 2, 4, 8)
pilot2_onetx_dims = (2, 3, 2, 4, 8)
group = "pilot2_channelmul"
pilot2_train_inputs_dir = os.path.join(datadir, group, 'train_channelmul_orient0.csv')
pilot2_train_labels_dir = os.path.join(datadir, group, 'pilot2train_labels.csv')
pilot2_test_inputs_dir = os.path.join(datadir, group, 'test_channelmul_orient0.csv')
pilot2_test_labels_dir = os.path.join(datadir, group, 'pilot2test_labels.csv')

pilot2_s1_train_inputs_dir = os.path.join(datadir, group, 'train_channelmul_s1.csv')
pilot2_s1_test_inputs_dir = os.path.join(datadir, group, 'test_channelmul_s1.csv')

pilot2_o1_s1_train_inputs_dir = os.path.join(datadir, group, 'train_channelmul_orient0_s1.csv')
pilot2_o1_s1_test_inputs_dir = os.path.join(datadir, group, 'test_channelmul_orient0_s1.csv')

# all the directory files combined
raw_dirs = ("pilot1_rssi", rssi_dir_test, labels_dir_test, train_idx_dir_test, test_idx_dir_test, just_rssi_dims)
r_dirs = ("pilot1_r", r_dir, labels_dir, train_idx_dir, test_idx_dir, just_r_dims)
r3one_dirs = ("pilot1_r_3dim_one", r_dir, labels_dir, train_idx_dir, test_idx_dir, just_r_dims)
r3avg_dirs = ("pilot1_r_3dim_avg", r_dir, labels_dir, train_idx_dir, test_idx_dir, just_r_dims)
r2_dirs = ("pilot1_r_2dim", r_dir, labels_dir, train_idx_dir, test_idx_dir, just_r_dims)

rphi_dirs = ("pilot1_r_phi", rphi_dir, labels_dir, train_idx_dir, test_idx_dir, all_dims)
rphi3_dirs = ("pilot1_r_phi_3dim", rphi_dir, labels_dir, train_idx_dir, test_idx_dir, all_dims)
rphi2_dirs = ("pilot1_r_phi_2dim", rphi_dir, labels_dir, train_idx_dir, test_idx_dir, all_dims)

rphi_rel_dirs = ("pilot1_r_phi_rel", rphi_rel_dir, labels_dir, train_idx_dir, test_idx_dir, phi_rel_dims)
rphi3_rel_dirs = ("pilot1_r_phi_rel_3dim", rphi_rel_dir, labels_dir, train_idx_dir, test_idx_dir, phi_rel_dims)
rphi2_rel_dirs = ("pilot1_r_phi_rel_2dim", rphi_rel_dir, labels_dir, train_idx_dir, test_idx_dir, phi_rel_dims)

ab_dirs = ("pilot1_a_b", ab_dir, labels_dir, train_idx_dir, test_idx_dir, all_dims)
ab3_dirs = ("pilot1_a_b_3dim", ab_dir, labels_dir, train_idx_dir, test_idx_dir, all_dims)
ab2_dirs = ("pilot1_a_b_2dim", ab_dir, labels_dir, train_idx_dir, test_idx_dir, all_dims)

channelmul_dirs = ("pilot1_channelmul", channelmul_dir, labels_dir, train_idx_dir, test_idx_dir, all_dims)
channelmul3_dirs = ("pilot1_channelmul_3dim", channelmul_dir, labels_dir, train_idx_dir, test_idx_dir, all_dims)
channelmul2_dirs = ("pilot1_channelmul_2dim", channelmul_dir, labels_dir, train_idx_dir, test_idx_dir, all_dims)


pilot2_dirs = ("pilot2_channelmul_orient0", pilot2_train_inputs_dir, pilot2_train_labels_dir, pilot2_test_inputs_dir, pilot2_test_labels_dir, pilot2_oneorientation_dims)
pilot2_3dim_dirs = ("pilot2_channelmul_orient0_3dim", pilot2_train_inputs_dir, pilot2_train_labels_dir, pilot2_test_inputs_dir, pilot2_test_labels_dir, pilot2_oneorientation_dims)
pilot2_2dim_dirs = ("pilot2_channelmul_orient0_2dim", pilot2_train_inputs_dir, pilot2_train_labels_dir, pilot2_test_inputs_dir, pilot2_test_labels_dir, pilot2_oneorientation_dims)
pilot2_s1_dirs = ("pilot2_channelmul_orient0_s1", pilot2_o1_s1_train_inputs_dir, pilot2_train_labels_dir, pilot2_o1_s1_test_inputs_dir, pilot2_test_labels_dir, pilot2_onetx_dims)
pilot2_s1_3dim_dirs = ("pilot2_channelmul_orient0_s1_3dim", pilot2_o1_s1_train_inputs_dir, pilot2_train_labels_dir, pilot2_o1_s1_test_inputs_dir, pilot2_test_labels_dir, pilot2_onetx_dims)
pilot2_s1_2dim_dirs = ("pilot2_channelmul_orient0_s1_2dim", pilot2_o1_s1_train_inputs_dir, pilot2_train_labels_dir, pilot2_o1_s1_test_inputs_dir, pilot2_test_labels_dir, pilot2_onetx_dims)


# modify which datasets you want to run here

dirs = [rphi_rel_dirs, rphi3_rel_dirs, rphi2_rel_dirs,
        channelmul_dirs, channelmul3_dirs, channelmul2_dirs, 
        pilot2_dirs, pilot2_3dim_dirs, pilot2_2dim_dirs, 
        pilot2_s1_dirs, pilot2_s1_3dim_dirs, pilot2_s1_2dim_dirs]

# MODEL PARAMETERS EDIT HERE
num_epoch = 200
lr = 0.001

import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class LocalizationCNN(nn.Module):
    def __init__(self, input_channels=2, dataset_name=""):
        super().__init__()
        output_dim = 3 # location: (x, y, z)

        # SETUP FOR MODEL PARAMETERS
        ignore = False
        if dataset_name == "pilot1_rssi": 
            model_param = [128, 128, 512, 512]
            ignore = True
        elif dataset_name == "pilot1_r":
            model_param = [128, 256, 256, 128]
        elif dataset_name == "pilot1_r_3dim_one" or dataset_name == "pilot1_r_3dim_avg":
            model_param = [128, 128, 256, 128]
            ignore = True
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
        elif dataset_name == "pilot1_channel" or dataset_name == "pilot1_channeldiv" or dataset_name == "pilot1_channelmul":
            model_param = [256, 256, 512, 512]
        elif dataset_name == "pilot1_channel_3dim" or dataset_name == "pilot1_channeldiv_3dim":
            model_param = [64, 64, 256, 128]
        elif dataset_name == "pilot1_channelmul_3dim":
            model_param = [128, 128, 256, 256]
        elif dataset_name == "pilot1_channel_2dim":
            model_param = [128, 256, 256, 128]
        elif dataset_name.find("2dim") != -1:
            model_param = [128, 128, 256, 128]
            ignore = True
        else:
            model_param = [256, 256, 1024, 512]
        model_sz, model_sz2, fc1_sz, fc2_sz = model_param

        if dataset_name.find("2dim") != -1 or dataset_name == "pilot1_rssi":
            pool1_k_sz = 1
            pool2_k_sz = (1,1)
        elif dataset_name.find("rel") != -1:
            pool1_k_sz = 2
            pool2_k_sz = (1,2)
        else:
            pool1_k_sz = 2
            pool2_k_sz = (2,2)
        conv_k_sz = 3

        self.conv1 = nn.Conv2d(input_channels, model_sz, kernel_size=conv_k_sz, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(model_sz)
        self.conv2 = nn.Conv2d(model_sz, model_sz, kernel_size=conv_k_sz, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(model_sz)
        self.pool1 = nn.MaxPool2d(kernel_size=pool1_k_sz, stride=pool1_k_sz)
        self.conv3 = nn.Conv2d(model_sz, model_sz2, kernel_size=conv_k_sz, stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(model_sz2)
        self.conv4 = nn.Conv2d(model_sz2, model_sz2, kernel_size=conv_k_sz, stride=1, padding=1)
        self.bn4   = nn.BatchNorm2d(model_sz2)
        self.pool2 = nn.MaxPool2d(kernel_size=pool2_k_sz, stride=(1,1))
        self.adapt = nn.AdaptiveAvgPool2d((1,1))

        self.fc1 = nn.Linear(model_sz2, fc1_sz)
        self.fc2 = nn.Linear(fc1_sz, fc2_sz)
        self.out = nn.Linear(fc2_sz, output_dim)
    
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.2)

        self.relu = nn.ReLU()

        if ignore:
            self.conv3 = nn.Identity()
            self.bn3   = nn.Identity()
            self.conv4 = nn.Identity()
            self.bn4   = nn.Identity()

    def forward(self, x):
        # x: (B, C, 6, 32) aka (B, C, 3 tx pos * 2 ant, 4 sym * 8 pkt)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)

        x = self.adapt(x)

        x = self.flatten(x)

        x = self.relu(self.fc1(x)) 
        x = self.dropout(x) # NEW
        
        x = self.relu(self.fc2(x)) 
        x = self.dropout(x) # NEW
        
        return self.out(x)

class RSSIDataset():
    def __init__(self, inputs, labels):
        self.inputs = torch.from_numpy(inputs).float()
        self.labels = torch.from_numpy(labels).float() # / 60.0 EDIT HERE FOR WORLDVIEW

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

def split_dataset(datadir, rssi_dir, train_index_dir, test_index_dir, ratio=0.8):
    """random shuffle train/test set
    """
    data = pd.read_csv(rssi_dir)
    index = np.arange(len(data))
    random.shuffle(index)

    train_len = int(len(index) * ratio)
    train_index = np.array(index[:train_len])
    test_index = np.array(index[train_len:])

    np.savetxt(train_index_dir, train_index, fmt='%s')
    np.savetxt(test_index_dir, test_index, fmt='%s')

def train_model(train_loader, val_loader=None, epochs=10, lr=0.0001, dataset_name="", skip = True , input_channels=2, seed=42):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LocalizationCNN(input_channels=input_channels, dataset_name=dataset_name).to(device)
    criterion = nn.MSELoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1, eta_min=1e-6)

    hist = {"train_loss": [], "val_loss": [], "train_batch_losses": [], "val_batch_losses": []}
    
    safe_lr = f"{lr:.0e}"
    ckpt_dir = f"{logsdir}/ckpts"
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val = float("inf")
    patience = 20
    counter = 0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        epoch_train_batch_losses = []
        model.train()
        running_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            epoch_train_batch_losses.append(loss.item())
            running_loss += loss.item() * x.size(0)
        scheduler.step() # Add outside of loop (Tien suggestion)

        epoch_loss = running_loss / len(train_loader.dataset)
        hist["train_loss"].append(epoch_loss)
        hist["train_batch_losses"].append(epoch_train_batch_losses.copy())

        # ---- validation ----
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
            val_loss /= len(val_loader.dataset)
            hist["val_loss"].append(val_loss)
            hist["val_batch_losses"].append(epoch_val_batch_losses)

            # NEW
            if val_loss < best_val:
                best_epoch = epoch
                best_val = val_loss
                counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                counter += 1
            #if counter >= patience:
                #print(f"Early stopping at epoch {epoch}")
                #break
                
            if epoch % 10 == 0:
                print(f"Epoch {epoch:03d}: train={epoch_loss:.6f}, val={val_loss:.6f}")
        else:
            if epoch % 10 == 0:
                print(f"Epoch {epoch:03d}: train={epoch_loss:.6f}")
            hist["val_loss"].append(None)  # placeholder
            hist["val_batch_losses"].append(None)

        # Save checkpoint every 50 epochs
        # if epoch % 50 == 0:
        #     ckpt_path = (
        #         f"{ckpt_dir}/loc_{dataset_name}_lr{safe_lr}_epoch{epoch}.pth"
        #     )
        #     torch.save(model.state_dict(), ckpt_path)
        #     print(f"Saved checkpoint: {ckpt_path}")

    return model, hist

def test_model(model, test_loader, device=None, save_csv=False, csv_path="predictions.csv"):
    """
    Evaluate a trained model on a test set.

    Args:
        model: Trained Localization model
        test_loader: DataLoader for test dataset
        device: torch.device (default: cuda if available)
        save_csv: bool, whether to save predictions
        csv_path: path to save predictions if save_csv=True

    Returns:
        predictions: numpy array of shape (num_samples, 3)
        targets: numpy array of shape (num_samples, 3)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            pred = pred

            predictions.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())

    predictions = np.vstack(predictions)
    targets = np.vstack(targets)

    mse_per_sample = np.mean((predictions - targets) ** 2, axis=1)
    mean_mse = np.mean(mse_per_sample)
    
    errors = np.linalg.norm(predictions - targets, axis=1)
    mean_error = np.mean(errors)

    median_error = np.median(errors)

    # Find large errors
    threshold = 30
    high_error_idx = np.where(errors > threshold)[0]
    
    print(f"\nNumber of samples with error > {threshold}: {len(high_error_idx)}")
    
    for idx in high_error_idx:
        print(f"\nSample {idx}")
        print(f"  True: {targets[idx]}")
        print(f"  Pred: {predictions[idx]}")
        print(f"  Error: {errors[idx]:.2f}")

    if save_csv:
        df = pd.DataFrame({
            "x_pred": predictions[:, 0],
            "y_pred": predictions[:, 1],
            "z_pred": predictions[:, 2],
            "x_true": targets[:, 0],
            "y_true": targets[:, 1],
            "z_true": targets[:, 2],
            "error": errors
        })
        df.to_csv(csv_path, index=False)
        print(f"Saved predictions and errors to {csv_path}")

    print(f"Mean Euclidean error: {mean_error:.4f}")
    print(f"Mean MSE: {mean_mse:.6f}")
    print(f"Median Error: {median_error:.6f}")
    return predictions, targets, errors
    
if __name__ == "__main__":
    set_seed(42)
    
    model_variants = {
        "baseline": False#,
        #"residual": True
    }
    mean_errors = []
    #cdf_errors = {"baseline": {}, "residual": {}}
    cdf_errors = {"baseline": {}}
    
    for dataset_name, inputs_dir, labels_dir, train_index_dir, test_index_dir, dims in dirs:
        if not os.path.exists(train_index_dir) or not os.path.exists(test_index_dir):
            split_dataset(datadir, inputs_dir, train_index_dir, test_index_dir, ratio=0.8)

        if dataset_name.startswith("pilot2"):
            train_inputs_dir = inputs_dir
            train_labels_dir = labels_dir
            test_inputs_dir = train_index_dir
            test_labels_dir = test_index_dir
            inputs = pd.read_csv(train_inputs_dir)
            labels = pd.read_csv(train_labels_dir)
    
            test_input = pd.read_csv(test_inputs_dir)
            test_label = pd.read_csv(test_labels_dir)

            # BEGIN MODIFY #########################################################################
            N_COMPLEX, N_TX, N_ANT, N_SYM, N_PKT = dims
    
            N = inputs.shape[0]
            N2 = test_input.shape[0]
            print("train INPUT shape: ", inputs.shape)
            print("test INPUT shape: ", test_input.shape)
            
            N_FEAT = N_TX * N_SYM * N_ANT * N_PKT
            
            # Update N_FEAT dynamically for 2-channel complex or 3-channel features
            if N_COMPLEX == 2:
                X = inputs.iloc[:, :N_FEAT].values + 1j * inputs.iloc[:, N_FEAT:].values
                X2 = test_input.iloc[:, :N_FEAT].values + 1j * test_input.iloc[:, N_FEAT:].values
            elif N_COMPLEX == 3:
                N_FEAT = N_COMPLEX * N_TX * N_SYM * N_PKT  # 3 channels: r1, r2, dphi
                X = inputs.values
            else:
                X = inputs.values
                X2 = input.values
            
            if N_COMPLEX != 3:
                X = X.reshape(N, N_TX, N_ANT, N_SYM, N_PKT)
                X2 = X2.reshape(N2, N_TX, N_ANT, N_SYM, N_PKT)
                # (N, 4, 6, 2, 4, 8)
            
                if dataset_name == "pilot2_channelmul_orient0_s1_3dim" or dataset_name == "pilot2_channelmul_orient0_3dim":
                    sym_idx = 0
                    X = X[:, :, :, sym_idx:sym_idx+1, :]
                    X2 = X2[:, :, :, sym_idx:sym_idx+1, :]
                    N_SYM = 1
            
                elif dataset_name == "pilot1_r_3dim_avg":
                    X = X.mean(axis=4, keepdims=True)
                    X2 = X2.mean(axis=4, keepdims=True)
                    N_SYM = 1
            
                elif dataset_name == "pilot2_channelmul_orient0_s1_2dim" or dataset_name == "pilot2_channelmul_orient0_2dim":
                    X = X.mean(axis=3, keepdims=True)
                    X2 = X2.mean(axis=3, keepdims=True)
                    N_SYM = 1
                    X = X.mean(axis=4, keepdims=True)
                    X2 = X2.mean(axis=4, keepdims=True)
                    N_PKT = 1
            
                # flatten per-orientation spatial dims
                # (N, 6, 32)
                X = X.reshape(N, N_TX * N_ANT, N_SYM * N_PKT)
                X2 = X2.reshape(N2, N_TX * N_ANT, N_SYM * N_PKT)
            
                if N_COMPLEX == 2:
                    inputs_cnn = np.stack([X.real, X.imag], axis=1)
                    test_inputs_cnn = np.stack([X2.real, X2.imag], axis=1)
                    # (N, 2, 6, 32)
                else:
                    inputs_cnn = X[:, None, :, :, :]
                    test_inputs_cnn = X2[:, None, :, :, :]
                    # (N, 1, 6, 32)
            # END MODIFY ###########################################################################
    
            test_inputs = test_inputs_cnn
            test_labels = test_label.values
            test_dataset = RSSIDataset(test_inputs, test_labels)
            test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
            train_inputs = inputs_cnn
            train_labels = labels.values
            train_dataset = RSSIDataset(train_inputs, train_labels)
        else:
            train_index = np.loadtxt(train_index_dir, dtype=int)
            test_index = np.loadtxt(test_index_dir, dtype=int)
    
            inputs = pd.read_csv(inputs_dir)
            labels = pd.read_csv(labels_dir)
        
            # BEGIN MODIFY #########################################################################
            N_COMPLEX, N_TX, N_SYM, N_ANT, N_PKT = dims
    
            N = inputs.shape[0]
            print("INPUT shape: ", inputs.shape)
            
            # Update N_FEAT dynamically for 2-channel complex or 3-channel features
            if N_COMPLEX == 2:
                N_FEAT = N_TX * N_SYM * N_ANT * N_PKT
                X = inputs.iloc[:, :N_FEAT].values + 1j * inputs.iloc[:, N_FEAT:].values
            elif N_COMPLEX == 3:
                N_FEAT = N_COMPLEX * N_TX * N_SYM * N_PKT  # 3 channels: r1, r2, dphi
                X = inputs.values
                X = X.reshape(N, N_COMPLEX, N_TX, N_SYM, N_PKT)  # (N, 3, 3, 4, 8)
    
                if dataset_name == "pilot1_r_phi_rel_3dim":
                    sym_idx = 0
                    X = X[:, :, :, sym_idx:sym_idx+1, :]  # (N, 3, 3, 1, 8)
                    N_SYM = 1
                elif dataset_name == "pilot1_r_phi_rel_2dim":
                    print("shrink dimension")
                    sym_idx = 0
                    X = X[:, :, :, sym_idx:sym_idx+1, :]  # (N, 3, 3, 1, 8)
                    N_SYM = 1
                    X = X.mean(axis=4, keepdims=True)
                    N_PKT = 1
            
                X = X.reshape(N, N_COMPLEX, N_TX, N_SYM * N_PKT)  # (N, 3, 3, 32) or (N,3,3,8)
                inputs_cnn = X
            else:
                N_FEAT = N_TX * N_SYM * N_ANT * N_PKT
                X = inputs.values
            
            if N_COMPLEX != 3:
                X = X.reshape(N, N_TX, N_SYM, N_ANT, N_PKT)  # (N, 3, 4, 2, 8)
            
                if dataset_name == "pilot1_a_b_3dim" or dataset_name == "pilot1_r_phi_3dim" or dataset_name == "pilot1_r_3dim_one":
                    sym_idx = 0
                    X = X[:, :, sym_idx:sym_idx+1, :, :]
                    N_SYM = 1
                elif dataset_name == "pilot1_r_3dim_avg":
                    X = X.mean(axis=2, keepdims=True)
                    N_SYM = 1
                elif dataset_name == "pilot1_r_2dim":
                    X = X.mean(axis=2, keepdims=True)
                    N_SYM = 1
                    X = X.mean(axis=4, keepdims=True)
                    N_PKT = 1
                elif dataset_name == "pilot1_channeldiv_3dim" or dataset_name == "pilot1_channelmul_3dim":
                    sym_idx = 0
                    X = X[:, :, sym_idx:sym_idx+1, :, :]
                    N_SYM = 1
                elif dataset_name == "pilot1_channeldiv_2dim" or dataset_name == "pilot1_channelmul_2dim":
                    sym_idx = 0
                    X = X[:, :, sym_idx:sym_idx+1, :, :]
                    N_SYM = 1
                    X = X.mean(axis=4, keepdims=True)
                    N_PKT = 1
                elif dataset_name == "pilot1_a_b_2dim" or dataset_name == "pilot1_r_phi_2dim":
                    sym_idx = 0
                    X = X[:, :, sym_idx:sym_idx+1, :, :]
                    N_SYM = 1
                    X = X.mean(axis=4, keepdims=True)
                    N_PKT = 1
            
                X = X.transpose(0, 1, 3, 2, 4)  # (N, 3, 2, 4, 8)
                X = X.reshape(N, N_TX * N_ANT, N_SYM * N_PKT)  # (N, 6, 32)
            
                if N_COMPLEX == 2:
                    inputs_cnn = np.stack([X.real, X.imag], axis=1)  # (N, 2, 6, 32)
                else:
                    inputs_cnn = X[:, None, :, :] # (N, 1, 6, 32)
            # END MODIFY ###########################################################################
    
            test_inputs = inputs_cnn[test_index]
            test_labels = labels.iloc[test_index].reset_index(drop=True).values
            test_dataset = RSSIDataset(test_inputs, test_labels)
            test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
            train_inputs = inputs_cnn[train_index]
            train_labels = labels.iloc[train_index].reset_index(drop=True).values
            train_dataset = RSSIDataset(train_inputs, train_labels)
        
        for seed in range(42, 42+5):
            g = torch.Generator()
            g.manual_seed(seed)
            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, generator=g)

            history_list = []
            label_list   = []
            results_dir = f"{plotsdir}"
            
            print(f"\n==== Seed {seed}: Training on {dataset_name} dataset with " + 
                  f"LR = {lr} on {num_epoch} epochs ====\n")
            model, hist = train_model(train_loader, val_loader=test_loader,
                                      epochs=num_epoch, lr=lr, dataset_name=dataset_name, 
                                      skip=False, input_channels=N_COMPLEX, seed=seed)
            
            csv_name = f"{logsdir}/predictions/pred_{dataset_name}_{seed}.csv"
            
            predictions, targets, errors = test_model(model, test_loader, save_csv = True, csv_path = csv_name);
            cdf_errors["baseline"][f"{dataset_name}_{seed}"] = errors

            xy_save_path = (f"{results_dir}/{dataset_name}/seed_{seed}/"
                f"diff_{dataset_name}_{seed}.png")
            
            plot_xy_prediction_lines(predictions, targets, train_xy=train_labels,
                                     title=f"Prediction difference w/ target - {dataset_name} data", 
                                     save_path=xy_save_path)
            
            plot_error_histogram(errors, 
                                 save_path=f"{results_dir}/{dataset_name}/testerr.png")

            mean_error = float(np.mean(errors))
            mean_errors.append({
                "dataset": dataset_name,
                "dimensions": f"{N} x {N_COMPLEX} x {N_TX} x {N_ANT} x {N_SYM} x {N_PKT}",
                "seed": seed,
                "mean_euclidean_error": mean_error
            })
            
            history_list.append(hist)
            label_list.append(f"lr={lr}_{num_epoch}ep")
            
            plot_results(history_list, label_list, dataset_name, save_dir = f"{results_dir}/{dataset_name}")
            plot_epoch_variability(history_list, label_list, dataset_name, save_dir = f"{results_dir}/{dataset_name}")
    plot_multiple_cdfs(cdf_errors["baseline"], save_path=f"{plotsdir}/cnn_cdf_overlay.png")
    #plot_multiple_cdfs(cdf_errors["residual"], save_path=f"{plotsdir}/residual/cdf_overlay.png")

    # Convert results to DataFrame
    results_df = pd.DataFrame(mean_errors)
    results_df["mean_euclidean_error"] = results_df["mean_euclidean_error"].map(lambda x: f"{x:.3f}")
    # Print clean table
    print(f"\n===== MODEL PERFORMANCE TABLE (Mean Errors) with LR = {lr} on {num_epoch} epochs =====")
    print(results_df.to_string(index=False, justify='left',
        formatters={
            "dataset":   lambda x: f"{x:<24}",
            "dimensions":lambda x: f"{x:<25}",
            "mean_error":lambda x: f"{x:<6.3f}",
        }
    ))
    
    # Save it
    results_df.to_csv(f"{logsdir}/cnn_mean_errors.csv", index=False)
    print(f"\nSaved table to {logsdir}/cnn_mean_errors.csv")
    #plot_mean_error_line(results_df, save_dir = f"{plotsdir}")