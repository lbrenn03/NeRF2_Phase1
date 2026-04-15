"""
F2 Receiver — Offline Processing
Loads a pre-captured .mat file and outputs nerf_<tag>.mat matching the MATLAB final_data struct layout.

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

parser = argparse.ArgumentParser(description='NeRF2 offline receiver')
parser.add_argument('--input',  default='mimo_tx1_1.5_3.5_0.mat', help='Input .mat file')
parser.add_argument('--output', default='processed.mat',        help='Output .mat file')
args = parser.parse_args()

# ==============================================================================
# 1.  CONFIGURATION
# ==============================================================================
Fs           = 500e3
CaptureTime  = 2.0
TotalSamples = round(Fs * CaptureTime)

SPS     = 5
ROLLOFF = 0.25
SPAN    = 6

NUM_PACKETS    = 120
MIN_PEAK_DIST  = 7000
MIN_PROM_FRAC  = 0.0
pn_bits_S1 = sio.loadmat('waveform_STTD.mat')['bits_S1'].ravel().astype(np.uint8)[6:]
pn_bits_S2 = sio.loadmat('waveform_STTD.mat')['bits_S2'].ravel().astype(np.uint8)[6:]



def qpsk_modulate(bits, phase_offset=np.pi / 4):
    assert len(bits) % 2 == 0
    b = bits.reshape(-1, 2)
    GRAY_MAP = {(0,0): 0, (0,1): 1, (1,1): 2, (1,0): 3}
    gray_idx = np.array([GRAY_MAP[tuple(bb)] for bb in b], dtype=int)
    return np.exp(1j * (phase_offset + np.pi / 2 * gray_idx))

def qpsk_demodulate(syms, phase_offset=np.pi / 4):
    candidates = np.exp(1j * (phase_offset + np.pi / 2 * np.arange(4)))
    dists      = np.abs(syms[:, None] - candidates[None, :])
    idx        = np.argmin(dists, axis=1)
    gray_to_bits = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.uint8)
    return gray_to_bits[idx].ravel()


def rrc_rx_filter(y, h, sps, ref_syms=None):
    """
    Matched filtering with optional fractional timing synchronization.
    
    y: Received signal
    h: RRC filter coefficients
    sps: Samples per symbol
    ref_syms: (Optional) Preamble symbols used to find the best sampling phase.
    """
    # 1. Perform the matched filtering at the full sampling rate
    yf = np.convolve(y, h, mode='same')
    
    if ref_syms is None:
        # Fallback to default behavior if no reference is provided
        return yf[::sps]

    best_phase = 0
    max_corr_sq = -1
    
    # 2. Search for the optimal fractional phase (k)
    # We iterate through all possible starting samples within one symbol period
    for k in range(sps):
        candidate = yf[k::sps]
        
        # Cross-correlate candidate symbols with the known preamble
        # We use 'valid' to find the peak within the sequence
        correlation = np.convolve(candidate, np.conj(ref_syms[::-1]), mode='valid')
        peak_val = np.max(np.abs(correlation)**2)
        
        if peak_val > max_corr_sq:
            max_corr_sq = peak_val
            best_phase = k
            
    # 3. Return the downsampled signal using the best phase found
    return yf[best_phase::sps]


def coarse_cfo_estimate(x, fs):
    x4   = x ** 4
    n    = len(x4)
    nfft = 1 << (n - 1).bit_length()
    X4   = np.fft.fft(x4, n=nfft)
    freqs = np.fft.fftfreq(nfft, d=1/fs)
    mag  = np.abs(X4)
    mag[0] = 0.0
    return float(freqs[int(np.argmax(mag))]) / 4.0


def xcorr(x, y):
    x = np.asarray(x, dtype=complex)
    y = np.asarray(y, dtype=complex)
    corr = sps_sig.correlate(x, y, mode='full', method='auto')
    lags = sps_sig.correlation_lags(len(x), len(y), mode='full')
    return corr, lags


def fit_to(arr, n):
    if len(arr) >= n:
        return arr[:n]
    return np.pad(arr, (0, n - len(arr)))


def kmeans_cluster_means(syms):
    pts = np.column_stack([syms.real, syms.imag])
    centroids, _ = kmeans2(pts, 4, iter=20, minit='points', missing='warn')
    return centroids[:, 0] + 1j * centroids[:, 1]


def compute_evm(syms):
    try:
        pts = np.column_stack([syms.real, syms.imag])
        centroids, labels = kmeans2(pts, 4, iter=20, minit='points', missing='warn')
        errors = pts - centroids[labels]
        return float(np.mean(np.sum(errors**2, axis=1)))
    except Exception:
        return float('NaN')


def detect_packets(rx, ref_sig):
    corrVals = sps_sig.correlate(rx, ref_sig, mode='full', method='auto')
    lags     = sps_sig.correlation_lags(len(rx), len(ref_sig), mode='full')
    corrMag  = np.abs(corrVals)
    min_prom = MIN_PROM_FRAC * np.max(corrMag)
    pks_locs, _ = sps_sig.find_peaks(corrMag, distance=MIN_PEAK_DIST, prominence=min_prom)
    pks_vals    = corrMag[pks_locs]
    sort_idx    = np.argsort(pks_vals)[::-1]
    pks_vals    = pks_vals[sort_idx]
    pks_locs    = pks_locs[sort_idx]
    print(f"corrVals {corrVals[:10]}")
    return corrVals, corrMag, lags, pks_locs, pks_vals


def collect_valid_packets(lags, pks_locs, pks_vals, n_pkt, buf_len):
    valid = []
    for k in range(len(pks_locs)):
        lag  = int(lags[pks_locs[k]])
        s_py = lag
        e_py = s_py + n_pkt
        if s_py >= 0 and e_py <= buf_len and pks_vals[k] > 0:
            valid.append({'startIdx': s_py, 'peakVal': float(pks_vals[k]), 'peakTime': s_py / Fs})
        if len(valid) >= NUM_PACKETS:
            break
    return valid


def process_packets(valid_packets, rxBuffer, ref_syms, pn_bits):
    NUM_REF_SYMS = len(ref_syms)
    num_p       = len(valid_packets)
    all_syms_ant1 = np.zeros((NUM_REF_SYMS, num_p), dtype=complex)
    all_syms_ant2 = np.zeros((NUM_REF_SYMS, num_p), dtype=complex)
    results = []

    for p, pkt in enumerate(valid_packets):
        s = pkt['startIdx'] - 1000#- 10000
        extraction_len = N_PKT - (SPAN // 2 * SPS) + 2000#+ 20000

        pkt_ant1 = rxBuffer[s:s+extraction_len, 0]
        pkt_ant2 = rxBuffer[s:s+extraction_len, 1]

        est_cfo = coarse_cfo_estimate(pkt_ant1, Fs)
        t_vec   = np.arange(len(pkt_ant2)) / Fs
        cfo_vec = np.exp(-1j * 2 * np.pi * est_cfo * t_vec)

        # Print the normalized frequency offset
        normalized_cfo = est_cfo / Fs
        print(f"CFO: {est_cfo:.2f} Hz")
        print(f"Phase rotation per symbol: {normalized_cfo * 360:.2f} degrees")

        clean_ant1 = pkt_ant1 * cfo_vec
        clean_ant2 = pkt_ant2 * cfo_vec

        if (len(pkt_ant1) == extraction_len and len(pkt_ant2) == extraction_len):
            
        
            mf_ant1 = rrc_rx_filter(clean_ant1, H_rrc, SPS, ref_syms=ref_syms)
            mf_ant2 = rrc_rx_filter(clean_ant2, H_rrc, SPS, ref_syms=ref_syms)

            fine_corr1, fine_lags1 = xcorr(mf_ant1, ref_syms)
            fine_idx1              = int(np.argmax(np.abs(fine_corr1)))
            los_phase_ref         = np.angle(fine_corr1[fine_idx1])
            phase_anchor          = np.exp(-1j * los_phase_ref)

            # In your packet detection block:
            peak_val = np.max(np.abs(fine_corr1))
            mean_val = np.mean(np.abs(fine_corr1))
            print(f"Correlation Peak-to-Average Ratio: {peak_val / mean_val:.2f}")

            fine_corr2, fine_lags2 = xcorr(mf_ant2, ref_syms)
            fine_idx2              = int(np.argmax(np.abs(fine_corr2)))

            mf_ant1_anchored = mf_ant1 * phase_anchor
            mf_ant2_anchored = mf_ant2 * phase_anchor

            fine_timing_offset1 = int(fine_lags1[fine_idx1])
            start_idx1 = max(0, min(fine_timing_offset1, len(mf_ant1) - NUM_REF_SYMS))
            end_idx1   = start_idx1 + NUM_REF_SYMS
            fine_timing_offset2 = int(fine_lags2[fine_idx2])
            start_idx2 = max(0, min(fine_timing_offset2, len(mf_ant2) - NUM_REF_SYMS))
            end_idx2   = start_idx2 + NUM_REF_SYMS

            syms_ant1 = mf_ant1_anchored[start_idx1:end_idx1]
            syms_ant2 = mf_ant2_anchored[start_idx2:end_idx2]

            symbol_indices_ant1 = range(len(syms_ant1))#np.where(np.abs(syms_ant1) > 0.5 * np.mean(np.abs(syms_ant1)))[0]
            symbol_indices_ant2 = range(len(syms_ant2))#np.where(np.abs(syms_ant2) > 0.5 * np.mean(np.abs(syms_ant2)))[0]

            syms_ant1 = syms_ant1[symbol_indices_ant1]
            syms_ant2 = syms_ant2[symbol_indices_ant2]

            # if p < 5:  # First 5 packets
            #     # Plot symbol magnitudes vs symbol index
            #     plt.figure(figsize=(10, 4))
            #     plt.plot(np.arange(len(syms_ant1)), np.abs(syms_ant1), marker='o', linestyle='-', label='Antenna 1')
            #     plt.plot(np.arange(len(syms_ant2)), np.abs(syms_ant2), marker='x', linestyle='-', label='Antenna 2')
            #     plt.xlabel('Symbol Index within Packet')
            #     plt.ylabel('Magnitude')
            #     plt.title(f'Packet {p} Symbol Magnitudes')
            #     plt.grid(True)
            #     plt.legend()
            #     plt.tight_layout()
            #     plt.show()

            all_syms_ant1[:, p] = fit_to(syms_ant1, NUM_REF_SYMS)
            all_syms_ant2[:, p] = fit_to(syms_ant2, NUM_REF_SYMS)
        

            # 1. Extract the symbols
            syms_to_demod1 = mf_ant1[start_idx1:end_idx1]
            syms_to_demod2 = mf_ant2[start_idx2:end_idx2]

            # Check for phase drift across the packet
            first_sym_phase = np.angle(syms_to_demod1[0] * np.conj(ref_syms[0]))
            last_sym_phase = np.angle(syms_to_demod1[-1] * np.conj(ref_syms[len(syms_to_demod1)-1]))
            print(f"Phase Drift across preamble: {np.degrees(last_sym_phase - first_sym_phase):.2f} degrees")
            p_signal = np.mean(np.abs(syms_to_demod1)**2)
            p_noise = np.abs(syms_to_demod1[0])**2 # First sample might be noise
            print(f"Estimated SNR (rough): {10 * np.log10(p_signal / p_noise):.2f} dB")

            # We compare the first N symbols of our extracted window to the known reference symbols
            num_ref = min(len(syms_to_demod1), len(ref_syms))
            phase_corr1 = np.angle(np.sum(syms_to_demod1[:num_ref] * np.conj(ref_syms[:num_ref])))
            phase_corr2 = np.angle(np.sum(syms_to_demod2[:num_ref] * np.conj(ref_syms[:num_ref])))

            # 3. Demodulate
            bits_out1 = qpsk_demodulate(syms_to_demod1 * np.exp(-1j * phase_corr1))
            bits_out2 = qpsk_demodulate(syms_to_demod2 * np.exp(-1j * phase_corr2))

            print(f"Truth Bits: {pn_bits[:20]}")
            print(f"RX Bits:    {bits_out1[:20]}")

            # 4. The BER calculation
            L = min(len(pn_bits), len(bits_out1))
        
            n_err1 = int(np.sum(pn_bits[:L] != bits_out1[:L]))
            n_err2 = int(np.sum(pn_bits[:L] != bits_out2[:L]))

            ber1 = n_err1 / L
            ber2 = n_err2 / L

            csi1 = np.mean(syms_ant1 / ref_syms[symbol_indices_ant1])
            csi2 = np.mean(syms_ant2 / ref_syms[symbol_indices_ant2])
            if p == 0:
                print(f"CSI1 {csi1}  CSI2 {csi2}")
            results.append({
                'ber1':      ber1,
                'ber2':      ber2,
                'cfo':       est_cfo,
                'csi1':      csi1,
                'csi2':      csi2,
                'rel_phase': np.angle(csi2),
                'peakVal':   pkt['peakVal'],
                'peakTime':  pkt['peakTime'],
            })

    return all_syms_ant1, all_syms_ant2, results


def cluster_packets(all_syms_ant1, all_syms_ant2):
    num_p = all_syms_ant1.shape[1]
    cluster_means_ant1 = np.full((4, num_p), np.nan, dtype=complex)
    cluster_means_ant2 = np.full((4, num_p), np.nan, dtype=complex)
    for p in range(num_p):
        try:
            cluster_means_ant1[:, p] = kmeans_cluster_means(all_syms_ant1[:, p])
        except Exception:
            pass
        try:
            cluster_means_ant2[:, p] = kmeans_cluster_means(all_syms_ant2[:, p])
        except Exception:
            pass
    return cluster_means_ant1, cluster_means_ant2


def plot_constellations(all_syms_ant1, all_syms_ant2, valid_packets, label):
    sample_idxs = range(min(len(valid_packets), 10))
    n_cols = max(len(sample_idxs), 2)
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))
    if n_cols == 1:
        axes = axes.reshape(2, 1)
    for si, p in enumerate(sample_idxs):
        syms1 = all_syms_ant1[:, p]
        syms2 = all_syms_ant2[:, p]
        ax = axes[0, si]
        ax.scatter(syms1.real, syms1.imag, s=8, c='b', alpha=0.3)
        ax.grid(True); ax.set_aspect('equal')
        ax.set_title(f'{label} Pkt {p+1} | Ant1 | EVM={compute_evm(syms1):.3f}', fontsize=8)
        ax.set_xlabel('I'); ax.set_ylabel('Q')
        ax = axes[1, si]
        ax.scatter(syms2.real, syms2.imag, s=8, c='r', alpha=0.3)
        ax.grid(True); ax.set_aspect('equal')
        ax.set_title(f'{label} Pkt {p+1} | Ant2 | EVM={compute_evm(syms2):.3f}', fontsize=8)
        ax.set_xlabel('I'); ax.set_ylabel('Q')
    for si in range(len(sample_idxs), n_cols):
        axes[0, si].set_visible(False)
        axes[1, si].set_visible(False)
    fig.tight_layout()


def print_summary(label, results):
    ber1_arr = np.array([r['ber1'] for r in results])
    ber2_arr = np.array([r['ber2'] for r in results])
    cfo_arr  = np.array([r['cfo']  for r in results])
    print(f"\n================ {label} SUMMARY ================")
    print(f"Total packets processed: {len(results)}")
    print(f"\nAntenna 1:")
    print(f"  Mean BER:    {np.mean(ber2_arr):.6f}")
    print(f"  Median BER:  {np.median(ber2_arr):.6f}")
    print(f"  Min BER:     {np.min(ber2_arr):.6f}")
    print(f"  Max BER:     {np.max(ber2_arr):.6f}")
    print(f"\nCFO Statistics:")
    print(f"  Mean CFO:    {np.mean(cfo_arr):.1f} Hz")
    print(f"  Std CFO:     {np.std(cfo_arr):.1f} Hz")
    print("=" * (len(label) + 27) + "\n")


# ==============================================================================
# 2.  BUILD REFERENCE SIGNALS
# ==============================================================================
H_rrc = sio.loadmat("rrcTx_coeffs.mat")["coeffs"].ravel()
# H_rrc = sio.loadmat("rrcTx_coeffs.mat")["coeffs"]['Numerator'].ravel()[0].ravel().astype(np.float64)
print(f"\nRRC coeffs: sum={np.sum(H_rrc):.6f} (should be ≈5.0 for RX)")
print(f"RRC energy: {np.sum(np.abs(H_rrc)**2):.6f}")

ref_syms_S1  = qpsk_modulate(pn_bits_S1)
ref_syms_S2  = qpsk_modulate(pn_bits_S2)
NUM_REF_SYMS = len(ref_syms_S1)

ref_sig_S1 = sio.loadmat('waveform_STTD.mat')['waveform_S1'].ravel()
ref_sig_S2 = sio.loadmat('waveform_STTD.mat')['waveform_S2'].ravel()

# And verify what qpsk_modulate produces matches ref_syms
ref_syms_check = qpsk_modulate(pn_bits_S1)
print("ref_syms_S1[:5]:", ref_syms_S1[:5])
print("modulated check[:5]:", ref_syms_check[:5])
print("match:", np.allclose(ref_syms_S1, ref_syms_check))

GROUP_DELAY = (SPAN * SPS) // 2
ref_sig_S1  = ref_sig_S1[GROUP_DELAY:]
ref_sig_S2  = ref_sig_S2[GROUP_DELAY:]

N_PKT = len(ref_sig_S1)
print(f"Reference signal: {N_PKT} samples | {NUM_REF_SYMS} symbols")
print(len(ref_sig_S2))
# ==============================================================================
# 3.  LOAD IQ DATA
# ==============================================================================
print(f"\nLoading {args.input} ...")
mat      = sio.loadmat(args.input, squeeze_me=False)
raw_data = mat['raw_data']
iq_data  = raw_data['iq'][0, 0]
rx_pos   = raw_data['rx_pos'][0, 0].flatten().astype(float)
tx_pos   = raw_data['tx_pos'][0, 0].flatten().astype(float)

rxBuffer = iq_data.astype(np.complex128)
rxBuffer = rxBuffer - np.mean(rxBuffer, axis=0)

if rxBuffer.shape[0] < TotalSamples:
    print(f"WARNING: Incomplete capture ({rxBuffer.shape[0]} < {TotalSamples} samples)")


# ==============================================================================
# 4.  PACKET DETECTION
# ==============================================================================
print("Cross-correlation for packet detection...")

rx1 = rxBuffer[:, 0]
rx2 = rxBuffer[:, 1]

corrVals_S1, corrMag_S1, lags_S1, pks_locs_S1, pks_vals_S1 = detect_packets(rx2, ref_sig_S1)
corrVals_S2, corrMag_S2, lags_S2, pks_locs_S2, pks_vals_S2 = detect_packets(rx2, ref_sig_S2)

valid_packets_S1 = collect_valid_packets(lags_S1, pks_locs_S1, pks_vals_S1, N_PKT, rxBuffer.shape[0])
valid_packets_S2 = collect_valid_packets(lags_S2, pks_locs_S2, pks_vals_S2, N_PKT, rxBuffer.shape[0])

if not valid_packets_S1:
    sys.exit("No valid S1 packets found above threshold.")
if not valid_packets_S2:
    sys.exit("No valid S2 packets found above threshold.")

print(f"Found {len(valid_packets_S1)} valid S1 packets")
print(f"Found {len(valid_packets_S2)} valid S2 packets")
print(f"S1 start indices: {[p['startIdx'] for p in valid_packets_S1]}")
print(f"S2 start indices: {[p['startIdx'] for p in valid_packets_S2]}\n")

# ==============================================================================
# 5.  CORRELATION PLOTS
# ==============================================================================
for label, lags, corrMag, valid_packets in [
    ('S1', lags_S1, corrMag_S1, valid_packets_S1),
    ('S2', lags_S2, corrMag_S2, valid_packets_S2),
]:
    causal_mask = (lags >= 0) & (lags <= (TotalSamples - N_PKT))
    causal_lags = lags[causal_mask]
    causal_corr = corrMag[causal_mask]
    causal_time = causal_lags / Fs
    causal_norm = causal_corr / np.max(causal_corr)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(causal_time, causal_norm, color=[0.2, 0.45, 0.8], linewidth=0.8, label='|Xcorr|')
    for k, pkt in enumerate(valid_packets):
        t_pk    = pkt['startIdx'] / Fs
        closest = int(np.argmin(np.abs(causal_time - t_pk)))
        ph      = float(causal_norm[closest])
        ax.plot([t_pk, t_pk], [0, ph], 'r-', linewidth=1.4)
        ax.plot(t_pk, ph, 'ro', markersize=6, markerfacecolor='r')
        ax.text(t_pk, min(ph + 0.05, 1.12), f'P{k+1}', fontsize=7, ha='center', color='r')
    ax.axhline(0.75, color='k', linestyle='--', linewidth=1, label='75% threshold')
    ax.set_xlim([0, CaptureTime]); ax.set_ylim([0, 1.15])
    ax.grid(True); ax.set_xlabel('Time (s)'); ax.set_ylabel('Normalised |Xcorr|')
    ax.set_title(f'{label} Cross-Correlation vs Time — {len(valid_packets)} Packets Detected')
    ax.legend(loc='upper right')
    fig.tight_layout()

# ==============================================================================
# 6.  PER-PACKET PROCESSING
# ==============================================================================
print("Processing S1 packets...")
all_syms_ant1_S1, all_syms_ant2_S1, results_S1 = process_packets(valid_packets_S1, rxBuffer, ref_syms_S1, pn_bits_S1)

print("Processing S2 packets...")
all_syms_ant1_S2, all_syms_ant2_S2, results_S2 = process_packets(valid_packets_S2, rxBuffer, ref_syms_S2, pn_bits_S2)

# ==============================================================================
# 7.  CLUSTERING
# ==============================================================================
print("\nComputing cluster means...")
cluster_means_ant1_S1, cluster_means_ant2_S1 = cluster_packets(all_syms_ant1_S1, all_syms_ant2_S1)
cluster_means_ant1_S2, cluster_means_ant2_S2 = cluster_packets(all_syms_ant1_S2, all_syms_ant2_S2)

# ==============================================================================
# 8.  CONSTELLATION PLOTS
# ==============================================================================
# plot_constellations(all_syms_ant1_S1, all_syms_ant2_S1, valid_packets_S1, 'S1')
# plot_constellations(all_syms_ant1_S2, all_syms_ant2_S2, valid_packets_S2, 'S2')

# ==============================================================================
# 9.  SUMMARY STATISTICS
# ==============================================================================
print_summary('S1', results_S1)
print_summary('S2', results_S2)

# ==============================================================================
# 10.  SAVE RESULTS
# ==============================================================================
BER_THRESHOLD = 0.001

def filter_by_ber(results, syms_ant1, syms_ant2, threshold=BER_THRESHOLD):
    """Return results and symbol arrays keeping only packets with BER1 <= threshold."""
    keep = [i for i, r in enumerate(results) if r['ber1'] <= threshold]
    n_orig = len(results)
    print(f"  BER filter: keeping {len(keep)}/{n_orig} packets (BER1 ≤ {threshold})")
    filtered_results = [results[i] for i in keep]
    filtered_ant1    = syms_ant1[:, keep] #if keep else np.zeros((syms_ant1.shape[0], 0), dtype=complex)
    filtered_ant2    = syms_ant2[:, keep] #if keep else np.zeros((syms_ant2.shape[0], 0), dtype=complex)
    return filtered_results, filtered_ant1, filtered_ant2, len(keep)

print("\nApplying BER filter before saving...")
results_S1_f, all_syms_ant1_S1_f, all_syms_ant2_S1_f, num_pkts = filter_by_ber(
    results_S1, all_syms_ant1_S1, all_syms_ant2_S1)
results_S2_f, all_syms_ant1_S2_f, all_syms_ant2_S2_f, num_pkts = filter_by_ber(
    results_S2, all_syms_ant1_S2, all_syms_ant2_S2)

plot_constellations(all_syms_ant1_S1, all_syms_ant2_S1, range(num_pkts), 'S1')
plot_constellations(all_syms_ant1_S2, all_syms_ant2_S2, range(num_pkts), 'S2')

# Re-cluster on the filtered symbol sets
cluster_means_ant1_S1_f, cluster_means_ant2_S1_f = cluster_packets(all_syms_ant1_S1_f, all_syms_ant2_S1_f)
cluster_means_ant1_S2_f, cluster_means_ant2_S2_f = cluster_packets(all_syms_ant1_S2_f, all_syms_ant2_S2_f)

def make_results_struct(results):
    return {
        'ber1':     np.array([r['ber1']     for r in results]).reshape(1, -1),
        'cfo':      np.array([r['cfo']      for r in results]).reshape(1, -1),
        'peakVal':  np.array([r['peakVal']  for r in results]).reshape(1, -1),
        'peakTime': np.array([r['peakTime'] for r in results]).reshape(1, -1),
    }

final_data = {
    'results_S1':             np.array([make_results_struct(results_S1_f)]),
    'results_S2':             np.array([make_results_struct(results_S2_f)]),
    'num_packets_S1':         np.array([[len(results_S1_f)]], dtype=float),
    'num_packets_S2':         np.array([[len(results_S2_f)]], dtype=float),
    'all_syms_ant1_S1':       all_syms_ant1_S1_f,
    'all_syms_ant2_S1':       all_syms_ant2_S1_f,
    'all_syms_ant1_S2':       all_syms_ant1_S2_f,
    'all_syms_ant2_S2':       all_syms_ant2_S2_f,
    'cluster_means_ant1_S1':  cluster_means_ant1_S1_f,
    'cluster_means_ant2_S1':  cluster_means_ant2_S1_f,
    'cluster_means_ant1_S2':  cluster_means_ant1_S2_f,
    'cluster_means_ant2_S2':  cluster_means_ant2_S2_f,
    'rx_pos':                 rx_pos.reshape(1, -1),
    'tx_pos':                 tx_pos.reshape(1, -1),
    'freq':                   np.array([[915e6]]),
    'tx_bf_angle':            np.array([[np.nan]]),
    'rx_orientation':         np.array([[np.nan]]),
    'tx_mimo':                np.array([[np.nan]]),
}

sio.savemat(args.output, {'final_data': final_data})
print(f"Results saved → {args.output}")

plt.show()
