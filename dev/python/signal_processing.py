
"""
F2 Receiver — Offline Processing
Equivalent to the MATLAB NeRF2 receiver script.

Loads a pre-captured .mat file containing:
  raw_data.iq      : (1000000, 2) complex double  [antenna 1, antenna 2]
  raw_data.rx_pos  : (1, 3)  receiver XYZ position
  raw_data.tx_pos  : (1, 3)  transmitter XYZ position

Outputs: nerf_<tag>.mat  matching the MATLAB final_data struct layout.

Usage:
    python nerf2_receiver.py --input raw_capture.mat --output nerf_0_7.mat
"""

import argparse
import sys
import numpy as np
import scipy.io as sio
import scipy.signal as sps_sig
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2

# ==============================================================================
# 0.  CLI
# ==============================================================================
parser = argparse.ArgumentParser(description='NeRF2 offline receiver')
# parser.add_argument('--input',  default='../pilot_data/channelCoherenceTest_raw.mat', help='Input .mat file')
parser.add_argument('--input',  default='../pilot_data/p1_tx2_6_9.mat', help='Input .mat file')
parser.add_argument('--output', default='processed.mat',    help='Output .mat file')
args = parser.parse_args()

# ==============================================================================
# 1.  CONFIGURATION  (must match TX and MATLAB script exactly)
# ==============================================================================
Fs           = 500e3
CaptureTime  = 2.0                          # seconds
TotalSamples = round(Fs * CaptureTime)       # 15 000 000

SPS     = 5
ROLLOFF = 0.25
SPAN    = 6

NUM_PACKETS      = 8
INTER_PKT_TIME   = 0.05                      # seconds (TX wait between packets)
MIN_PEAK_DIST    = round(INTER_PKT_TIME * Fs) # samples
MIN_PROM_FRAC    = 0.6                       # fraction of global max

pn_bits = [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,1,1,1,1,1,1,0,1,1,1,0,0,0,1,0,0,1,1,1,1,1,0,0,0,1,1,0,0,1,1,1,1,1,0,1,0,1,1,0,0,1,0,1,1,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,1,1,0,0,1,0,0,1,1,0,1,0,0,0,0,1,0,0,1,0,1,0,1,0,0,0,0,1,1,1,1,0,1,0,1,1,1,0,1,0,1,1,0,1,1,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,1,1,0,0,1,1,0,0,0,0,1,0,1,0,1,1,0,1,0,1,1,1,0,0,0,1,1,0,1,1,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,0,0,1,1,1,1,0,1,1,0,1,1,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,1,0,1,1,0,1,0,1,0,1,0,0,0,1,1,1,1,1,0,1,1,1,1,0,0,1,0,0,1,0,1,1,0,0,0,0,0,1,0,0,1,1,0,0,1,0,0,0,1,0,1,0,0,0,1,1,0,1,1,0,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,1,1,1,0,1,1,1,1,1,1,1,0,0,1,0,0,0,0,1,1,0,0,0,1,0,1,1,0,1,1,1,0,1,0,0,0,0,1,1,0,1,0,1,0,1,1,0,0,1,1,1,1,0,0,1,0,1,1,0,1,1,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,1,1,0,0,0,0,0,0,1,0,1,1,0,0,0,1,0,1,0,0,1,1,1,0,1,1,0,0,1,1,1,0,0,0,1,0,1,1,1,1,1,1,0,1,0,1,0,0,0,1,0,1,1,1,0,1,1,0,1,0,1,1,0,0,0,0,1,1,0,0,1,1,0,1,1,0,1,0,1,0,0,0,0,0,1,1,1,0,1,0,0,1,1,1,1,0,1,0,0,1,1,0,1,0,1,0,0,1,0,0,1,1,1,0,0,0,0,0,1,1,1,1,1,0,0,1,1,1,0,0,1,1,0,1,1,1,1,0,1,0,0,0,1,0,1,0,1,0,1,1,0,1,1,1,1,1,0,0,0,0,1,0,0,1,1,1,0,1,0,0,0,1,1,1,0,1,0,1,1,1,1,1,0,1,1,0,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,1,0,0,1,0,1,0,1,1,0,0,0,1,1,1,0,0,1,1,1,1,1,1,1,0,1,1,0,0,0,0,1,0,0,0,1,1,0,1,0,0,1,1,1,0,0,1,0,0,1,1,1,1,0,0,0,0,1,1,0,1,1,1,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,0,1,1,1,1,1,0,1,0,0,1,0,0,1,0,1,0,0,0,0,0,0,1,1,0,1,0,0,0,1,1,0,0,1,0,1,1,1,0,1,0,0,1,0,1,1,0,1,0,0,0,1,0,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1,0,1,0,0,1,0,0,0,1,1,0,0,0,0,1,1,1,0,1,1,0,1,1,1,1,0,0,0,0,0,1,0,1,1,1,0,0,1,0,1,0,1,1,1,0,0,1,1,1,0,1,1,1,0,1,1,1,0,0,1,1,0,0,1,1,1,0,1,0,1,0,1,1,1,0,1,1,1,1,0,1,1,0,0,1,0,1,0,0,0,1,0,0,1,1,0,1,1,0,0,0,1,0,0,0,0,1,1,1,0,0,1,0,1,1,1,1,1,0,0,1,0,1,0,0,1,1,0,0,1,1,0,0,1,0,1,0,1,0,1,0,0,1,1,1,1,1,1,0,0,1,1,0,0,0,1,1,0,1,0,1,1,1,1,0,0,1,1,0,1,0,1,1,0,1,0,0,1,1,0,0,0,1,0,0,1,0,1,1,1,0,0,0,0,1,0,1,1,1,1,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,0,1,0,0,0,0,0,1,0,1,0,1,0,0,1,0,1,1,1,1,0,0,0,1,0,1,0,1,1,1,1,0,1,1,1,0,1,0,1,0,0,1,1,0,1,1,1,0,0,1,0,0,0,1,1,1,0,0,0]

ref_bits = pn_bits.copy()
if len(ref_bits) % 2 != 0:               # match MATLAB: pad to even length
    ref_bits = np.append(ref_bits, 0)


def qpsk_modulate(bits, phase_offset=np.pi / 4):
    """
    QPSK modulation matching MATLAB comm.QPSKModulator:
      BitInput=true, PhaseOffset=pi/4, default Gray coding.

    Gray map (MSB first):
      00 → index 0 → phase = pi/4   → ( 1+1j)/√2
      01 → index 1 → phase = 3pi/4  → (-1+1j)/√2
      11 → index 2 → phase = 5pi/4  → (-1-1j)/√2
      10 → index 3 → phase = 7pi/4  → ( 1-1j)/√2
    """
    assert len(bits) % 2 == 0
    b = bits.reshape(-1, 2)
    GRAY_MAP = {
        (0,0): 0,
        (0,1): 1,
        (1,1): 2,
        (1,0): 3,
    }

    gray_idx = np.array([GRAY_MAP[tuple(bb)] for bb in b], dtype=int)
    return np.exp(1j * (phase_offset + np.pi / 2 * gray_idx))


def qpsk_demodulate(syms, phase_offset=np.pi / 4):
    """
    QPSK demodulation matching MATLAB comm.QPSKDemodulator:
      BitOutput=true, PhaseOffset=pi/4.
    Returns uint8 bit array (MSB first per symbol).
    """
    candidates = np.exp(1j * (phase_offset + np.pi / 2 * np.arange(4)))
    dists      = np.abs(syms[:, None] - candidates[None, :])  # (N, 4)
    idx        = np.argmin(dists, axis=1)
    # Gray decode: 0→00, 1→01, 2→11, 3→10
    gray_to_bits = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.uint8)
    return gray_to_bits[idx].ravel()


def rrc_rx_filter(x, h, sps, span=SPAN, k=0):
    # Perform matched filtering and decimation
    y = sps_sig.upfirdn(h, x, up=1, down=sps)
    
    # Calculate group delay (15 samples / 5 sps = 3 symbols)
    delay_syms = (len(h) - 1) // (2 * sps) 
    
    # Remove the filter build-up but KEEP the extra samples at the end
    y = y[delay_syms + k :]
    
    return y # Do not truncate to Ns here


# Build reference signal
H_rrc   = sio.loadmat("rrcTx_coeffs.mat")["coeffs"].ravel()
print(f"\nRRC coeffs: sum={np.sum(H_rrc):.6f} (should be ≈5.0 for RX)")
print(f"RRC energy: {np.sum(np.abs(H_rrc)**2):.6f}")

ref_syms = qpsk_modulate(ref_bits)
NUM_REF_SYMS = len(ref_syms)

reference_sig = sio.loadmat("reference_sig.mat")["reference_sig"].ravel()

# Trim transients
GROUP_DELAY = (SPAN * SPS) // 2  # 15 samples
reference_sig = reference_sig[GROUP_DELAY:-GROUP_DELAY]

N_PKT = len(reference_sig)                 # packet length in samples

print(f"Reference signal : {N_PKT} samples | {NUM_REF_SYMS} symbols")

# ==============================================================================
# 3.  LOAD PRE-CAPTURED IQ DATA
# ==============================================================================
print(f"\nLoading {args.input} ...")
mat = sio.loadmat(args.input, squeeze_me=False)

# MATLAB structs arrive as object arrays; drill in with [0,0]
raw_data = mat['raw_data']
iq_data  = raw_data['iq'][0, 0]            # shape (15000000, 2), complex128
rx_pos   = raw_data['rx_pos'][0, 0].flatten().astype(float)
tx_pos   = raw_data['tx_pos'][0, 0].flatten().astype(float)

rxBuffer = iq_data.astype(np.complex128)   # ensure complex double
# Remove the mean (DC component) from each antenna
rxBuffer = rxBuffer - np.mean(rxBuffer, axis=0)

if rxBuffer.shape[0] < TotalSamples:
    print(f"WARNING: Incomplete capture ({rxBuffer.shape[0]} < {TotalSamples} samples)")


# ==============================================================================
# 4.  IQ IMBALANCE COMPENSATION  (Adaptive WL-LMS, matching MATLAB)
# ==============================================================================
# Model:  y[n] = x[n] + w * conj(x[n])
# Update: w    = w  - mu * conj(y[n]) * x[n]
#
# Performance note: running sample-by-sample over 15M points in Python would be
# prohibitively slow.  The weight converges within a few thousand samples
# (hardware imbalance is a fixed, static property).  We therefore:
#   1. Run the LMS loop on the first CONV_LEN samples to obtain converged w.
#   2. Apply the static correction  y = x + w_conv * conj(x)  to the full buffer.
# This matches MATLAB's steady-state output to high accuracy.

CONV_LEN = 50_000   # samples needed for LMS convergence (empirically generous)
MU_IQ    = 1e-4     # LMS step size (MATLAB default for this compensator)

def iq_compensate(x, mu=MU_IQ, conv_len=CONV_LEN):
    # Normalize the step size by signal power
    conv_seg = x[:conv_len]
    power_estimate = np.mean(np.abs(conv_seg)**2)
    mu_normalized = mu / (power_estimate + 1e-10)  # Power-normalized step size
    
    w = 0.0 + 0.0j
    for xi in conv_seg:
        yi = xi + w * np.conj(xi)
        w  = w - mu_normalized * np.conj(yi) * xi
    
    return x + w * np.conj(x)

# print("IQ imbalance compensation...")
# rxBuffer[:, 0] = iq_compensate(rxBuffer[:, 0])
# rxBuffer[:, 1] = iq_compensate(rxBuffer[:, 1])

# ==============================================================================
# 5.  PACKET DETECTION 
# ==============================================================================

print("Cross-correlation for packet detection...")

rx1 = rxBuffer[:, 0]
corrVals = sps_sig.correlate(rx1, reference_sig, mode='full', method='auto')
lags = sps_sig.correlation_lags(len(rx1), len(reference_sig), mode='full')
corrMag   = np.abs(corrVals)

# --- Peak detection with spacing and prominence constraints ---
min_prom = MIN_PROM_FRAC * np.max(corrMag)
pks_locs, props = sps_sig.find_peaks(
    corrMag,
    distance=MIN_PEAK_DIST,
    prominence=min_prom
)
pks_vals = corrMag[pks_locs]

# Sort peaks by strength (descending), as in MATLAB
sort_idx  = np.argsort(pks_vals)[::-1]
pks_vals  = pks_vals[sort_idx]
pks_locs  = pks_locs[sort_idx]

# --- Collect valid packets ---
valid_packets = []
for k in range(len(pks_locs)):
    lag  = int(lags[pks_locs[k]])
    s_py = lag                          # 0-indexed Python start (MATLAB: s = lag+1)
    e_py = s_py + N_PKT

    if s_py >= 0 and e_py <= rxBuffer.shape[0] and pks_vals[k] > 0:
        valid_packets.append({
            'startIdx': s_py,
            'peakVal':  float(pks_vals[k]),
            'peakTime': s_py / Fs,
        })

    if len(valid_packets) >= NUM_PACKETS:
        break

if not valid_packets:
    sys.exit("❌  No valid packets found above threshold.")

print(f"✅  Found {len(valid_packets)} valid packets\n")

# ==============================================================================
# 5b.  CORRELATION PLOT  (Figure 10 equivalent)
# ==============================================================================
causal_mask  = (lags >= 0) & (lags <= (TotalSamples - N_PKT))
causal_lags  = lags[causal_mask]
causal_corr  = corrMag[causal_mask]
causal_time  = causal_lags / Fs
causal_norm  = causal_corr / np.max(causal_corr)

fig10, ax10 = plt.subplots(figsize=(14, 4))
ax10.plot(causal_time, causal_norm, color=[0.2, 0.45, 0.8], linewidth=0.8,
          label='|Xcorr|')

# Mark each detected packet
for k, pkt in enumerate(valid_packets):
    t_pk    = pkt['startIdx'] / Fs
    closest = int(np.argmin(np.abs(causal_time - t_pk)))
    ph      = float(causal_norm[closest])
    ax10.plot([t_pk, t_pk], [0, ph], 'r-', linewidth=1.4)
    ax10.plot(t_pk, ph, 'ro', markersize=6, markerfacecolor='r')
    ax10.text(t_pk, min(ph + 0.05, 1.12), f'P{k + 1}',
              fontsize=7, ha='center', color='r')

ax10.axhline(0.75, color='k', linestyle='--', linewidth=1,
             label='75% threshold')
ax10.set_xlim([0, CaptureTime])
ax10.set_ylim([0, 1.15])
ax10.grid(True)
ax10.set_xlabel('Time (s)')
ax10.set_ylabel('Normalised |Xcorr|')
ax10.set_title(f'Cross-Correlation vs Time — {len(valid_packets)} Packets Detected')
ax10.legend(loc='upper right')
fig10.tight_layout()

# ==============================================================================
# 6.  PER-PACKET PROCESSING
# ==============================================================================

def coarse_cfo_estimate(x, fs):
    """
    4th-power FFT CFO estimation for QPSK (matches MATLAB
    comm.CoarseFrequencyCompensator with Modulation='QPSK').

    Raises signal to 4th power to remove QPSK modulation, finds the spectral
    peak, divides by 4 to recover CFO in Hz.
    """
    x4    = x ** 4
    n     = len(x4)
    nfft = 1 << (n-1).bit_length()
    X4 = np.fft.fft(x4, n=nfft)
    freqs = np.fft.fftfreq(nfft, d=1/fs)
    mag   = np.abs(X4)
    mag[0] = 0.0                # suppress DC
    peak_idx = int(np.argmax(mag))
    return float(freqs[peak_idx]) / 4.0   # divide by 4 for QPSK


def xcorr(x, y, normalize=False):
    """Cross-correlation matching MATLAB xcorr(...) with optional normalization."""
    x = np.asarray(x, dtype=complex)
    y = np.asarray(y, dtype=complex)
    
    # Compute correlation
    corr = sps_sig.correlate(x, y, mode='full', method='auto')
    
    if normalize:
        # Normalization matching MATLAB 'normalized' option
        lags = sps_sig.correlation_lags(len(x), len(y), mode='full')
        
        norm_factors = np.zeros(len(corr))
        for i, lag in enumerate(lags):
            if lag >= 0:
                x_segment = x[lag:min(lag+len(y), len(x))]
                y_segment = y[:min(len(y), len(x)-lag)]
            else:
                x_segment = x[:min(len(x), len(y)+lag)]
                y_segment = y[-lag:min(-lag+len(x), len(y))]
            
            energy_x = np.sum(np.abs(x_segment)**2)
            energy_y = np.sum(np.abs(y_segment)**2)
            norm_factors[i] = np.sqrt(energy_x * energy_y)
        
        # Avoid division by zero
        norm_factors[norm_factors == 0] = 1.0
        corr = corr / norm_factors
    
    lags = sps_sig.correlation_lags(len(x), len(y), mode='full')
    return corr, lags

def fit_to(arr, n):
        if len(arr) >= n:
            return arr[:n]
        return np.pad(arr, (0, n - len(arr)))

# Storage — matching MATLAB's pre-allocated size of 500 symbols per packet.
# Analysis: after RRC rx-filter, mf has exactly N_PKT//SPS = 512 samples.
# Both branch-1 and branch-2 of the MATLAB symbol-extraction if-else are
# unreachable for non-zero timing offsets when mf length == NUM_REF_SYMS.
# The else fallback (mf[span : end-span]) yields exactly 500 symbols, which
# is why MATLAB pre-allocates 500.  We replicate that behaviour here.
SYM_STORE = 500

num_p        = len(valid_packets)
all_syms_ant1 = np.zeros((SYM_STORE, num_p), dtype=complex)
all_syms_ant2 = np.zeros((SYM_STORE, num_p), dtype=complex)

results = []

# ==============================================================================
# 6.  PER-PACKET PROCESSING (NeRF2 Optimized with Phase-Referencing)
# ==============================================================================

for p, pkt in enumerate(valid_packets):
    s = pkt['startIdx']

    # --- Extraction & CFO Correction ---
    extraction_limit = min(s + N_PKT + (SPAN * SPS), rxBuffer.shape[0])
    pkt_ant1 = rxBuffer[s : extraction_limit, 0]
    pkt_ant2 = rxBuffer[s : extraction_limit, 1]

    est_cfo = coarse_cfo_estimate(pkt_ant1, Fs)
    t_vec   = np.arange(len(pkt_ant1)) / Fs
    cfo_vec = np.exp(-1j * 2 * np.pi * est_cfo * t_vec)

    clean_ant1 = pkt_ant1 * cfo_vec
    clean_ant2 = pkt_ant2 * cfo_vec

    # --- Matched filter ---
    mf_ant1 = rrc_rx_filter(clean_ant1, H_rrc, SPS)
    mf_ant2 = rrc_rx_filter(clean_ant2, H_rrc, SPS)

    # --- Fine Sync & LoS Phase Extraction ---
    # We correlate to find the strongest path (LoS)
    fine_corr1, fine_lags = xcorr(mf_ant1, ref_syms, normalize=False) 
    fine_idx              = int(np.argmax(np.abs(fine_corr1)))
    
    # This is the "Direct Path" phase we must zero out
    los_phase_ref = np.angle(fine_corr1[fine_idx]) 

    # --- Phase-Referencing (The NeRF2 Fix) ---
    # Rotate BOTH antennas by the LoS phase of Antenna 1
    # This anchors Ant1 to 0 rad and keeps the spatial delta on Ant2
    phase_anchor = np.exp(-1j * los_phase_ref)
    
    mf_ant1_anchored = mf_ant1 * phase_anchor
    mf_ant2_anchored = mf_ant2 * phase_anchor

    # --- Symbol Extraction ---
    fine_timing_offset = int(fine_lags[fine_idx])
    start_idx = max(0, min(fine_timing_offset, len(mf_ant1) - NUM_REF_SYMS))
    end_idx   = start_idx + NUM_REF_SYMS

    syms_ant1 = mf_ant1_anchored[start_idx:end_idx]
    syms_ant2 = mf_ant2_anchored[start_idx:end_idx]

    # --- CSI Fingerprint Extraction ---
    # CSI = Mean complex gain relative to ideal QPSK constellation
    # Since we anchored the phase, csi1.imag should be near 0.
    csi1 = np.mean(syms_ant1 / ref_syms[:len(syms_ant1)])
    csi2 = np.mean(syms_ant2 / ref_syms[:len(syms_ant2)])
    print("CSI1", csi1, "CSI2", csi2)

    if p < 5:  # First 5 packets
        print("start:", start_idx, ", end:", end_idx)
        # Plot symbol magnitudes vs symbol index
        plt.figure(figsize=(10, 4))
        plt.plot(np.arange(len(syms_ant1)), np.abs(syms_ant1), marker='o', linestyle='-', label='Antenna 1')
        plt.plot(np.arange(len(syms_ant2)), np.abs(syms_ant2), marker='x', linestyle='-', label='Antenna 2')
        plt.xlabel('Symbol Index within Packet')
        plt.ylabel('Magnitude')
        plt.title(f'Packet {p} Symbol Magnitudes')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Store anchored symbols for the final NeRF dataset
    all_syms_ant1[:, p] = fit_to(syms_ant1, SYM_STORE)
    all_syms_ant2[:, p] = fit_to(syms_ant2, SYM_STORE)

    # --- BER & Stats ---
    rx_bits1 = qpsk_demodulate(syms_ant1) # Already phase-aligned
    rx_bits2 = qpsk_demodulate(syms_ant2)
    
    L = min(len(ref_bits), len(rx_bits1))
    n_err1 = int(np.sum(ref_bits[:L] != rx_bits1[:L]))
    n_err2 = int(np.sum(ref_bits[:L] != rx_bits2[:L]))
    
    results.append({
        'ber1':     n_err1 / L,
        'cfo':      est_cfo,
        'csi1':     csi1, 
        'csi2':     csi2,
        'rel_phase': np.angle(csi2), # This is your spatial fingerprint
        'peakVal':  pkt['peakVal'],
        'peakTime': pkt['peakTime'],
    })

# ==============================================================================
# 7.  K-MEANS CLUSTER MEANS PER PACKET  (matching MATLAB section 7)
# ==============================================================================
cluster_means_ant1 = np.full((4, num_p), np.nan, dtype=complex)
cluster_means_ant2 = np.full((4, num_p), np.nan, dtype=complex)

def kmeans_cluster_means(syms):
    """K-means (k=4) using SciPy."""
    # Stack I and Q into a (N, 2) array
    pts = np.column_stack([syms.real, syms.imag])

    # kmeans2(data, k, iter, thresh, minit)
    # 'points' minit uses 4 random points from the data as initial centroids
    centroids, labels = kmeans2(pts, 4, iter=20, minit='points', missing='warn')

    # Convert centroids back to complex numbers
    means = centroids[:, 0] + 1j * centroids[:, 1]
    return means

print("\nComputing cluster means...")
for p in range(num_p):
    try:
        cluster_means_ant1[:, p] = kmeans_cluster_means(all_syms_ant1[:, p])
    except Exception:
        pass
    try:
        cluster_means_ant2[:, p] = kmeans_cluster_means(all_syms_ant2[:, p])
    except Exception:
        pass

# ==============================================================================
# 8.  DIAGNOSTIC CONSTELLATION FIGURE  (Figure 2 equivalent — 4 sample packets)
# ==============================================================================
sample_idxs = range(NUM_PACKETS)
n_cols = max(len(sample_idxs), 2)

fig2, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))
if n_cols == 1:
    axes = axes.reshape(2, 1)

def compute_evm(syms):
    """EVM relative to SciPy k-means cluster centres."""
    try:
        pts = np.column_stack([syms.real, syms.imag])
        centroids, labels = kmeans2(pts, 4, iter=20, minit='points', missing='warn')

        # Calculate distance of each point to its assigned centroid
        assigned_centroids = centroids[labels]
        errors = pts - assigned_centroids
        evm = np.mean(np.sum(errors**2, axis=1))
        return float(evm)
    except Exception:
        return float('NaN')

for si, p in enumerate(sample_idxs):
    syms1 = all_syms_ant1[:, p]
    syms2 = all_syms_ant2[:, p]
    evm1  = compute_evm(syms1)
    evm2  = compute_evm(syms2)

    ax = axes[0, si]
    ax.scatter(syms1.real, syms1.imag, s=8, c='b', alpha=0.3)
    ax.grid(True)
    ax.set_aspect('equal')
    ax.set_title(f'Pkt {p + 1} | Ant1 | EVM={evm1:.3f}', fontsize=8)
    ax.set_xlabel('I')
    ax.set_ylabel('Q')

    ax = axes[1, si]
    ax.scatter(syms2.real, syms2.imag, s=8, c='r', alpha=0.3)
    ax.grid(True)
    ax.set_aspect('equal')
    ax.set_title(f'Pkt {p + 1} | Ant2 | EVM={evm2:.3f}', fontsize=8)
    ax.set_xlabel('I')
    ax.set_ylabel('Q')

# Hide any unused axes
for si in range(len(sample_idxs), n_cols):
    axes[0, si].set_visible(False)
    axes[1, si].set_visible(False)

fig2.tight_layout()

# ==============================================================================
# 9.  SUMMARY STATISTICS
# ==============================================================================
ber1_arr = np.array([r['ber1']    for r in results])
cfo_arr  = np.array([r['cfo']     for r in results])

print("\n================ SUMMARY STATISTICS ================")
print(f"Total packets processed: {num_p}")
print("\nAntenna 1:")
print(f"  Mean BER:    {np.mean(ber1_arr):.6f}")
print(f"  Median BER:  {np.median(ber1_arr):.6f}")
print(f"  Min BER:     {np.min(ber1_arr):.6f}")
print(f"  Max BER:     {np.max(ber1_arr):.6f}")
print("\nCFO Statistics:")
print(f"  Mean CFO:    {np.mean(cfo_arr):.1f} Hz")
print(f"  Std CFO:     {np.std(cfo_arr):.1f} Hz")
print("====================================================\n")

# ==============================================================================
# 10.  SAVE RESULTS  (matches MATLAB final_data struct layout exactly)
# ==============================================================================
# scipy.io.savemat replicates the nested struct by using a dict-of-dicts.
# Each scalar/array field is shaped to match MATLAB's column-vector convention.

results_struct = {
    'ber1':     ber1_arr.reshape(1, -1),
    'cfo':      cfo_arr.reshape(1, -1),
    'peakVal':  np.array([r['peakVal']  for r in results]).reshape(1, -1),
    'peakTime': np.array([r['peakTime'] for r in results]).reshape(1, -1),
}

final_data = {
    'results':            np.array([results_struct]),   # wrap in object array for struct
    'num_packets':        np.array([[num_p]], dtype=float),
    'all_syms_ant1':      all_syms_ant1,               # (500, num_packets) complex
    'all_syms_ant2':      all_syms_ant2,
    'cluster_means_ant1': cluster_means_ant1,          # (4, num_packets) complex
    'cluster_means_ant2': cluster_means_ant2,
    'rx_pos':             rx_pos.reshape(1, -1),
    'tx_pos':             tx_pos.reshape(1, -1),
    'freq':               np.array([[915e6]]),
    'tx_bf_angle':        np.array([[np.nan]]),
    'rx_orientation':     np.array([[np.nan]]),
    'tx_mimo':            np.array([[np.nan]]),
}

sio.savemat(args.output, {'final_data': final_data})
print(f"Results saved → {args.output}")

plt.show()
