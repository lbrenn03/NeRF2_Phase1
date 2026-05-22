# List of all plotting functions used by loc_model.py
import torch
import torch.nn as nn
import os
import pandas as pd
from torch.utils.data import DataLoader
import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

def plot_error_histogram(errors, save_path="error_histogram.png", bins=50):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    mean_error = np.mean(errors)

    plt.figure(figsize=(8,5))

    # Histogram
    plt.hist(errors, bins=bins, alpha=0.7, label="Error Distribution")

    # Mean line
    plt.axvline(mean_error, color='red', linestyle='--', linewidth=2,
                label=f"Mean Error = {mean_error:.3f}")

    # Labels and styling
    plt.xlabel("Euclidean Error")
    plt.ylabel("Count")
    plt.title("Distribution of Test Errors (with Mean)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()

def plot_results(history_list, label_list, dataset_name, save_dir="plots"):
    """
    Plot overlaid train + val loss curves for all models.
    """
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(9, 6))

    for hist, label in zip(history_list, label_list):
        epochs = range(1, len(hist["train_loss"]) + 1)

        # Training curve (solid)
        plt.plot(epochs, hist["train_loss"], label=f"{label} - train")

        # Validation curve (dashed)
        if hist["val_loss"][0] is not None:
            plt.plot(epochs, hist["val_loss"], linestyle="--", label=f"{label} - val")

    plt.title(f"Training & Validation Loss (Overlay) - {dataset_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(0,3)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "train_val_overlay.png"))
    plt.close()

def plot_epoch_variability(history_list, label_list, dataset_name, save_dir="plots"):
    """
    Plot within-epoch batch loss variability (min–max range) across all runs, for each epoch,
    and also plot all individual batch losses as very translucent points.
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(9, 6))
    
    skip_epochs = 5  # number of initial batches to ignore per epoch

    os.makedirs(save_dir, exist_ok=True)

    min_epochs = min(len(h["train_batch_losses"]) for h in history_list)

    num_epochs = min_epochs - skip_epochs
    if num_epochs <= 0:
        raise ValueError(f"skip_epochs={skip_epochs} >= min_epochs={min_epochs}")

    train_epoch_batches = [[] for _ in range(num_epochs)]
    val_epoch_batches   = [[] for _ in range(num_epochs)]

    for hist in history_list:
        for e in range(skip_epochs, min_epochs):
            idx = e - skip_epochs
            train_epoch_batches[idx].extend(hist["train_batch_losses"][e])
            val_epoch_batches[idx].extend(hist["val_batch_losses"][e])

    # Compute mean, min, max per epoch
    train_mean, train_min, train_max = [], [], []
    val_mean, val_min, val_max = [], [], []

    # Epoch numbers for x-axis
    epochs = np.arange(skip_epochs + 1, min_epochs + 1)  # 1-indexed

    plt.figure(figsize=(10, 6))

    for i in range(num_epochs):
        t_losses = np.array(train_epoch_batches[i])
        v_losses = np.array(val_epoch_batches[i])

        # Skip if empty (rare)
        if t_losses.size == 0 or v_losses.size == 0:
            continue

        # Scatter all individual batch points, translucent
        plt.scatter([epochs[i]]*len(t_losses), t_losses, color='blue', alpha=0.05, s=10)
        plt.scatter([epochs[i]]*len(v_losses), v_losses, color='green', alpha=0.05, s=10)

        # Compute stats
        train_mean.append(t_losses.mean())
        train_min.append(t_losses.min())
        train_max.append(t_losses.max())

        val_mean.append(v_losses.mean())
        val_min.append(v_losses.min())
        val_max.append(v_losses.max())

    # Convert lists to numpy arrays
    train_mean = np.array(train_mean)
    train_min  = np.array(train_min)
    train_max  = np.array(train_max)
    val_mean   = np.array(val_mean)
    val_min    = np.array(val_min)
    val_max    = np.array(val_max)

    # Plot mean ± min/max bands
    plt.plot(epochs, train_mean, color='blue', label='Train (mean)')
    plt.fill_between(epochs, train_min, train_max, color='blue', alpha=0.25, label='Train range')

    plt.plot(epochs, val_mean, linestyle='--', color='green', label='Val (mean)')
    plt.fill_between(epochs, val_min, val_max, color='green', alpha=0.25, label='Val range')

    plt.title(f"Within-Epoch Loss Variability – {dataset_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(save_dir, "train_val_within_epoch_range.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Saved plot to {save_path}")

def plot_multiple_cdfs(error_dict, save_path="cdf_overlay.png"):
    """
    error_dict : { "label1": errors1, "label2": errors2, ... }
    Uses absolute errors in meters.
    Each line gets a different auto-color.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8,5))

    for label, errors in error_dict.items():
        errors = np.asarray(errors)

        # compute CDF
        x = np.sort(errors)
        y = np.arange(1, len(x) + 1) / len(x)

        plt.plot(x, y, linewidth=2, label=label)

    plt.xlabel("Localization Error (meters)")
    plt.ylabel("CDF")
    plt.title("CDF of Localization Error (meters)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    # plt.xlim(0, 30)
    # plt.savefig(save_path + "_zoomed")
    plt.close()

def plot_mean_error_line(results_df, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(9, 5))
    # Ensure consistent dataset ordering
    dataset_order = results_df["dataset"].unique()

    for model_name, df_m in results_df.groupby("model"):
        df_m = df_m.set_index("dataset").loc[dataset_order].reset_index()

        plt.plot(
            df_m["dataset"],
            df_m["mean_error"],
            marker="o",
            label=model_name
        )

    plt.xlabel("Dataset")
    plt.ylabel("Mean Euclidean Error")
    plt.title("Mean Localization Error vs Dataset")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Model")
    plt.tight_layout()

    save_path = os.path.join(save_dir, "mean_error_line.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Saved plot to {save_path}")

def plot_xy_prediction_lines(predictions, targets, train_xy=None, max_points=None, title="XY Prediction vs Target", save_path="xy_pred_lines.png"):
    """
    Plot lines between target (x,y) and prediction (x,y) locations,
    optionally including training points.

    Args:
        predictions: numpy array of shape (N, 3)
        targets: numpy array of shape (N, 3)
        train_xy: numpy array of shape (M, 2) or (M, 3)
        max_points: int or None
    """
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    pred_xy = predictions[:, :2]
    target_xy = targets[:, :2]

    if train_xy is not None:
        train_xy = train_xy[:, :2]

    if max_points is not None:
        pred_xy = pred_xy[:max_points]
        target_xy = target_xy[:max_points]

    errors_xy = np.linalg.norm(pred_xy - target_xy, axis=1)

    vmin, vmax = 0, 10
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("Blues")

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)

    # --- NEW: plot training points ---
    if train_xy is not None:
        ax.scatter(
            train_xy[:, 0], train_xy[:, 1],
            c="lightgray",
            s=10,
            alpha=0.4,
            label="Train Points"
        )

    # Targets
    ax.scatter(
        target_xy[:, 0], target_xy[:, 1],
        marker="o",
        label="Target"
    )

    # Lines (target -> prediction)
    for (tx, ty), (px, py), err in zip(target_xy, pred_xy, errors_xy):
        dx = tx - px
        dy = ty - py
    
        ax.arrow(
            px, py,                # start at prediction
            dx, dy,                # vector toward target
            color=cmap(norm(err)),
            linewidth=0.5,
            alpha=0.9,
            length_includes_head=True,
            head_width=0.1,        # tweak based on your scale
            head_length=0.15
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.set_title(title)
    ax.axis("equal")
    ax.grid(True)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("XY Error Magnitude")

    ticks = cbar.get_ticks()
    ticks[-1] = vmax
    cbar.set_ticks(ticks)
    tick_labels = [str(int(tick)) for tick in ticks]
    tick_labels[-1] = f"{int(vmax)}+"
    cbar.set_ticklabels(tick_labels)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved XY prediction plot to: {save_path}")
    plt.close()