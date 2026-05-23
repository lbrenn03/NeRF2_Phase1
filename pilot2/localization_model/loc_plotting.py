"""
================================================================================
loc_plotting.py — Visualization Helpers for RF Localization CNN Pipeline
================================================================================

All plotting functions used by loc_train.py (and loc_cnn_pilot1.py).
Each function saves its output to disk and closes the figure to prevent
memory leaks from accumulated unclosed matplotlib figure objects.

Functions
---------
  plot_error_histogram       — histogram of per-sample Euclidean errors
  plot_results               — overlaid train/val loss curves per run
  plot_epoch_variability     — within-epoch batch loss scatter + mean/range band
  plot_multiple_cdfs         — CDF overlay across multiple dataset variants
  plot_mean_error_line       — mean error vs dataset name, one line per model
  plot_xy_prediction_lines   — arrow map of predicted vs true (x, y) positions

Usage
-----
  from loc_plotting import *

All functions call os.makedirs(..., exist_ok=True) on their output directory
before saving, so no directories need to be created in advance.

See README.md for full argument documentation.

Authors
-------
  Luke, Tien, and collaborators — Wireless Localization Research Group

Dependencies
------------
  NumPy, pandas, matplotlib, PyTorch (imported but used only for type compat)
================================================================================
"""

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
    """
    Plot a histogram of per-sample Euclidean errors with a mean marker.

    Args:
        errors    (np.ndarray): 1-D array of per-sample Euclidean errors in metres.
        save_path (str)       : Output file path. Parent directory is created if needed.
        bins      (int)       : Number of histogram bins. Default 50.

    Output: single PNG — error distribution with dashed mean line annotated in legend.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    mean_error = np.mean(errors)

    plt.figure(figsize=(8, 5))

    # Bar chart of error counts across the range of observed errors
    plt.hist(errors, bins=bins, alpha=0.7, label="Error Distribution")

    # Vertical dashed line marking the mean error for quick reference
    plt.axvline(mean_error, color='red', linestyle='--', linewidth=2,
                label=f"Mean Error = {mean_error:.3f}")

    plt.xlabel("Euclidean Error")
    plt.ylabel("Count")
    plt.title("Distribution of Test Errors (with Mean)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()  # release figure memory


def plot_results(history_list, label_list, dataset_name, save_dir="plots"):
    """
    Overlay train and validation loss curves for one or more training runs.

    Training curves are drawn as solid lines; validation curves as dashed lines.
    All runs share the same axes for direct comparison.

    Args:
        history_list (list[dict]): List of hist dicts returned by train_model.
                                   Each dict must have keys "train_loss" and "val_loss".
        label_list   (list[str]) : Legend label for each run (e.g. "lr=0.001_200ep").
        dataset_name (str)       : Inserted into the plot title.
        save_dir     (str)       : Directory where train_val_overlay.png is written.

    Output: <save_dir>/train_val_overlay.png
    """
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(9, 6))

    for hist, label in zip(history_list, label_list):
        epochs = range(1, len(hist["train_loss"]) + 1)

        # Solid line for training loss
        plt.plot(epochs, hist["train_loss"], label=f"{label} - train")

        # Dashed line for validation loss (skip if val was not used during training)
        if hist["val_loss"][0] is not None:
            plt.plot(epochs, hist["val_loss"], linestyle="--", label=f"{label} - val")

    plt.title(f"Training & Validation Loss (Overlay) - {dataset_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(0, 3)   # fixed y-axis so plots are comparable across datasets
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "train_val_overlay.png"))
    plt.close()  # release figure memory


def plot_epoch_variability(history_list, label_list, dataset_name, save_dir="plots"):
    """
    Show within-epoch batch loss variability across all runs.

    For each epoch (after an initial burn-in period), every individual batch
    loss is plotted as a translucent scatter point. A mean line and a shaded
    min-to-max band are overlaid on top. Training is shown in blue; validation
    in green.

    Args:
        history_list (list[dict]): List of hist dicts from train_model. Each dict
                                   must have keys "train_batch_losses" and
                                   "val_batch_losses" (lists-of-lists).
        label_list   (list[str]) : Currently unused; reserved for future multi-run
                                   labelling.
        dataset_name (str)       : Inserted into the plot title.
        save_dir     (str)       : Directory where the output PNG is written.

    Output: <save_dir>/train_val_within_epoch_range.png

    Notes:
        skip_epochs = 5 — the first 5 epochs are excluded to avoid the large
        initial loss values from distorting the y-axis scale.
    """
    os.makedirs(save_dir, exist_ok=True)

    skip_epochs = 5  # burn-in epochs to exclude from the variability plot

    # Find shortest run so all histories are aligned to the same epoch count
    min_epochs = min(len(h["train_batch_losses"]) for h in history_list)

    num_epochs = min_epochs - skip_epochs
    if num_epochs <= 0:
        raise ValueError(f"skip_epochs={skip_epochs} >= min_epochs={min_epochs}")

    # Pre-allocate one list per post-burn-in epoch to collect batch losses
    # across all runs (seeds) for aggregation
    train_epoch_batches = [[] for _ in range(num_epochs)]
    val_epoch_batches   = [[] for _ in range(num_epochs)]

    # Flatten all per-run batch losses into the per-epoch accumulators
    for hist in history_list:
        for e in range(skip_epochs, min_epochs):
            idx = e - skip_epochs  # 0-indexed position in post-burn-in arrays
            train_epoch_batches[idx].extend(hist["train_batch_losses"][e])
            val_epoch_batches[idx].extend(hist["val_batch_losses"][e])

    # Per-epoch summary statistics (mean, min, max) computed after aggregation
    train_mean, train_min, train_max = [], [], []
    val_mean,   val_min,   val_max   = [], [], []

    # 1-indexed epoch numbers for the x-axis (skip_epochs+1 … min_epochs)
    epochs = np.arange(skip_epochs + 1, min_epochs + 1)

    # NOTE: only one plt.figure() call here — the duplicate at the top of this
    # function was removed to prevent a memory leak from unclosed figures
    plt.figure(figsize=(10, 6))

    for i in range(num_epochs):
        t_losses = np.array(train_epoch_batches[i])
        v_losses = np.array(val_epoch_batches[i])

        # Guard against empty epochs (rare edge case)
        if t_losses.size == 0 or v_losses.size == 0:
            continue

        # Individual batch losses as translucent dots to show the raw spread
        plt.scatter([epochs[i]] * len(t_losses), t_losses, color='blue',  alpha=0.05, s=10)
        plt.scatter([epochs[i]] * len(v_losses), v_losses, color='green', alpha=0.05, s=10)

        # Accumulate summary stats for the mean/range band drawn after the loop
        train_mean.append(t_losses.mean())
        train_min.append(t_losses.min())
        train_max.append(t_losses.max())

        val_mean.append(v_losses.mean())
        val_min.append(v_losses.min())
        val_max.append(v_losses.max())

    # Convert to arrays for vectorised plotting
    train_mean = np.array(train_mean)
    train_min  = np.array(train_min)
    train_max  = np.array(train_max)
    val_mean   = np.array(val_mean)
    val_min    = np.array(val_min)
    val_max    = np.array(val_max)

    # Mean line + shaded min-to-max band for training losses
    plt.plot(epochs, train_mean, color='blue', label='Train (mean)')
    plt.fill_between(epochs, train_min, train_max, color='blue', alpha=0.25, label='Train range')

    # Mean line + shaded min-to-max band for validation losses
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
    plt.close()  # release figure memory

    print(f"Saved plot to {save_path}")


def plot_multiple_cdfs(error_dict, save_path="cdf_overlay.png"):
    """
    Overlay empirical CDFs of Euclidean localization error for multiple runs.

    Each entry in error_dict produces one labelled line. Lines are
    auto-coloured by matplotlib's default colour cycle.

    Args:
        error_dict (dict[str, np.ndarray]): Keys are run labels
                                            (e.g. "pilot1_a_b_42"); values
                                            are 1-D arrays of errors in metres.
        save_path  (str)                  : Output file path. Parent directory
                                            is created if needed.

    Output: single PNG — CDF of localization error in metres.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 5))

    for label, errors in error_dict.items():
        errors = np.asarray(errors)

        # Sort errors to build the empirical CDF: x = sorted errors, y = cumulative fraction
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
    # Zoomed version (uncomment to also save a 0–30 m view):
    # plt.xlim(0, 30)
    # plt.savefig(save_path + "_zoomed")
    plt.close()  # release figure memory


def plot_mean_error_line(results_df, save_dir="plots"):
    """
    Plot mean Euclidean error vs dataset name, with one line per model variant.

    Intended for comparing baseline vs residual (or other) model types across
    all datasets in a single chart.

    Args:
        results_df (pd.DataFrame): Must contain columns "dataset", "model",
                                   and "mean_error". Dataset order in the plot
                                   follows the order they first appear in the
                                   DataFrame.
        save_dir   (str)         : Directory where mean_error_line.png is written.

    Output: <save_dir>/mean_error_line.png

    Note: not called by default in loc_train.py (the call is commented out).
          Requires a "model" column not present in the default mean_errors list.
    """
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(9, 5))

    # Preserve the original dataset ordering rather than sorting alphabetically
    dataset_order = results_df["dataset"].unique()

    for model_name, df_m in results_df.groupby("model"):
        # Reindex to the canonical dataset order so all lines share the same x-axis
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
    plt.close()  # release figure memory

    print(f"Saved plot to {save_path}")


def plot_xy_prediction_lines(predictions, targets, train_xy=None, max_points=None,
                              title="XY Prediction vs Target", save_path="xy_pred_lines.png"):
    """
    Arrow map of predicted vs true (x, y) positions with error-coloured arrows.

    An arrow is drawn from each predicted position toward its true target.
    Arrow colour encodes the XY error magnitude on a blue colormap, saturating
    at vmax=10 m (shown as "10+" on the colorbar). Training sample locations
    are optionally shown as light grey background scatter points for spatial
    context.

    Args:
        predictions (np.ndarray, shape (N, 3)): Predicted (x, y, z) positions.
        targets     (np.ndarray, shape (N, 3)): Ground-truth (x, y, z) positions.
        train_xy    (np.ndarray, shape (M, 2 or 3)): Training positions for
                                                      background reference. Optional.
        max_points  (int or None): Cap the number of plotted samples. Useful for
                                   large test sets where arrow overlap is heavy.
        title       (str)        : Plot title.
        save_path   (str)        : Output file path. Parent directory is created
                                   if needed.

    Output: single PNG at save_path.

    Arrow convention: tail at prediction, head at true target — a longer arrow
    indicates a larger error. Colorbar saturates at 10 m; the top tick is
    labelled "10+" to make the saturation explicit.
    """
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Use only the (x, y) components; z is not plotted
    pred_xy   = predictions[:, :2]
    target_xy = targets[:, :2]

    if train_xy is not None:
        train_xy = train_xy[:, :2]  # drop z if present

    # Optionally limit the number of arrows to reduce clutter
    if max_points is not None:
        pred_xy   = pred_xy[:max_points]
        target_xy = target_xy[:max_points]

    # 2-D Euclidean error (ignores z) used to colour each arrow
    errors_xy = np.linalg.norm(pred_xy - target_xy, axis=1)

    # Colormap: 0 m (white/light blue) → 10 m (dark blue); saturates beyond vmax
    vmin, vmax = 0, 10
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("Blues")

    fig = plt.figure(figsize=(7, 7))
    ax  = fig.add_subplot(111)

    # Background: training sample locations for spatial reference
    if train_xy is not None:
        ax.scatter(
            train_xy[:, 0], train_xy[:, 1],
            c="lightgray", s=10, alpha=0.4,
            label="Train Points"
        )

    # True target positions as circular markers
    ax.scatter(target_xy[:, 0], target_xy[:, 1], marker="o", label="Target")

    # Draw one arrow per test sample: tail=prediction, head=true target
    for (tx, ty), (px, py), err in zip(target_xy, pred_xy, errors_xy):
        dx = tx - px  # displacement vector from prediction toward target
        dy = ty - py

        ax.arrow(
            px, py,                   # arrow starts at the predicted position
            dx, dy,                   # vector to the true target
            color=cmap(norm(err)),
            linewidth=0.5,
            alpha=0.9,
            length_includes_head=True,
            head_width=0.1,           # tune relative to the room scale
            head_length=0.15
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.set_title(title)
    ax.axis("equal")  # preserve spatial aspect ratio so distances read correctly
    ax.grid(True)

    # Colorbar: maps arrow colour back to error magnitude in metres
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("XY Error Magnitude")

    # Label the top colorbar tick as "10+" to communicate saturation clearly
    ticks = cbar.get_ticks()
    ticks[-1] = vmax
    cbar.set_ticks(ticks)
    tick_labels       = [str(int(tick)) for tick in ticks]
    tick_labels[-1]   = f"{int(vmax)}+"
    cbar.set_ticklabels(tick_labels)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved XY prediction plot to: {save_path}")
    plt.close()  # release figure memory