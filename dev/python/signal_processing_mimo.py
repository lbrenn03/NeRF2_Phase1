"""
MIMO Receiver — Offline Processing

Usage:
    python mimo_proc.py --input raw_capture.mat --output processed.mat
"""

import argparse
import sys
import numpy as np
import scipy.io as sio
import scipy.signal as sps_sig

parser = argparse.ArgumentParser(description='NeRF2 offline receiver')
parser.add_argument('--input',  default='example.mat', help='Input .mat file')
parser.add_argument('--output', default='processed_example.mat',             help='Output .mat file')
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

NUM_PACKETS   = 120
MIN_PEAK_DIST = 5000
MIN_PROM_FRAC = 0.0
BER_THRESHOLD = 0.001

pn_bits_S1 = sio.loadmat('waveform_STTD.mat')['bits_S1'].ravel().astype(np.uint8)[6:]
pn_bits_S2 = sio.loadmat('waveform_STTD.mat')['bits_S2'].ravel().astype(np.uint8)[6:]


# ==============================================================================
# 2.  HELPERS
# ==============================================================================
def qpsk_modulate(bits, phase_offset=np.pi / 4):
    assert len(bits) % 2 == 0
    b = bits.reshape(-1, 2)
    GRAY_MAP = {(0, 0): 0, (0, 1): 1, (1, 1): 2, (1, 0): 3}
    gray_idx = np.array([GRAY_MAP[tuple(bb)] for bb in b], dtype=int)
    return np.exp(1j * (phase_offset + np.pi / 2 * gray_idx))


def qpsk_demodulate(syms, phase_offset=np.pi / 4):
    candidates = np.exp(1j * (phase_offset + np.pi / 2 * np.arange(4)))
    dists = np.abs(syms[:, None] - candidates[None, :])
    idx = np.argmin(dists, axis=1)
    gray_to_bits = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.uint8)
    return gray_to_bits[idx].ravel()


def rrc_rx_filter(y, h, sps, ref_syms=None):
    yf = np.convolve(y, h, mode='same')
    if ref_syms is None:
        return yf[::sps]
    best_phase, max_corr_sq = 0, -1
    for k in range(sps):
        candidate = yf[k::sps]
        correlation = np.convolve(candidate, np.conj(ref_syms[::-1]), mode='valid')
        peak_val = np.max(np.abs(correlation) ** 2)
        if peak_val > max_corr_sq:
            max_corr_sq = peak_val
            best_phase = k
    return yf[best_phase::sps]


def coarse_cfo_estimate(x, fs):
    x4 = x ** 4
    n = len(x4)
    nfft = 1 << (n - 1).bit_length()
    X4 = np.fft.fft(x4, n=nfft)
    freqs = np.fft.fftfreq(nfft, d=1 / fs)
    mag = np.abs(X4)
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


def detect_packets(rx, ref_sig):
    corrVals = sps_sig.correlate(rx, ref_sig, mode='full', method='auto')
    lags = sps_sig.correlation_lags(len(rx), len(ref_sig), mode='full')
    corrMag = np.abs(corrVals)
    min_prom = MIN_PROM_FRAC * np.max(corrMag)
    pks_locs, _ = sps_sig.find_peaks(corrMag, distance=MIN_PEAK_DIST, prominence=min_prom)
    pks_vals = corrMag[pks_locs]
    sort_idx = np.argsort(pks_vals)[::-1]
    return corrVals, corrMag, lags, pks_locs[sort_idx], pks_vals[sort_idx]


def collect_valid_packets(lags, pks_locs, pks_vals, n_pkt, buf_len):
    valid = []
    for k in range(len(pks_locs)):
        lag = int(lags[pks_locs[k]])
        s_py = lag
        e_py = s_py + n_pkt
        if s_py >= 0 and e_py <= buf_len and pks_vals[k] > 0:
            valid.append({'startIdx': s_py, 'peakVal': float(pks_vals[k]), 'peakTime': s_py / Fs})
        if len(valid) >= NUM_PACKETS:
            break
    return valid


def process_packets(valid_packets, rxBuffer, ref_syms, pn_bits):
    NUM_REF_SYMS = len(ref_syms)
    results = []
    collected_syms_ant1 = []
    collected_syms_ant2 = []

    for pkt in valid_packets:
        # 1. BRAKE CHECK: Explicit Bounds Handling (Fixes Bug 5)
        s = pkt['startIdx'] - 1000
        extraction_len = N_PKT - (SPAN // 2 * SPS) + 2000
        
        if s < 0 or (s + extraction_len) > rxBuffer.shape[0]:
            continue

        pkt_ant1 = rxBuffer[s:s + extraction_len, 0]
        pkt_ant2 = rxBuffer[s:s + extraction_len, 1]

        # 2. CFO Correction
        est_cfo = coarse_cfo_estimate(pkt_ant1, Fs)
        t_vec = np.arange(extraction_len) / Fs
        cfo_vec = np.exp(-1j * 2 * np.pi * est_cfo * t_vec)
        clean_ant1 = pkt_ant1 * cfo_vec
        clean_ant2 = pkt_ant2 * cfo_vec

        mf_ant1 = rrc_rx_filter(clean_ant1, H_rrc, SPS, ref_syms=ref_syms)
        mf_ant2 = rrc_rx_filter(clean_ant2, H_rrc, SPS, ref_syms=ref_syms)

        fine_corr1, fine_lags1 = xcorr(mf_ant1, ref_syms)
        fine_idx1 = int(np.argmax(np.abs(fine_corr1)))
        start_idx = int(fine_lags1[fine_idx1])
        
        if start_idx < 0 or (start_idx + NUM_REF_SYMS) > len(mf_ant1):
            continue

        # Extract both antennas using the SAME temporal start point
        syms_ant1 = mf_ant1[start_idx : start_idx + NUM_REF_SYMS]
        syms_ant2 = mf_ant2[start_idx : start_idx + NUM_REF_SYMS]

        # 4. ROBUST PHASE ANCHOR: Bulk average instead of single tap
        # We calculate phase error of Ant1 and apply that rotation to BOTH
        phase_anchor_val = np.sum(syms_ant1 * np.conj(ref_syms))
        phase_corr1 = np.angle(phase_anchor_val)
        
        # Rotate both to maintain relative phase difference
        syms_ant1_rotated = syms_ant1 * np.exp(-1j * phase_corr1)
        syms_ant2_rotated = syms_ant2 * np.exp(-1j * phase_corr1)

        # 5. STRICT BER: No truncation allowed (Fixes Bug 4)
        bits_out1 = qpsk_demodulate(syms_ant1_rotated)
        # For BER 2, we must re-phase Ant2 to its own peak to check its data integrity
        phase_corr2 = np.angle(np.sum(syms_ant2 * np.conj(ref_syms)))
        bits_out2 = qpsk_demodulate(syms_ant2 * np.exp(-1j * phase_corr2))

        if len(bits_out1) != len(pn_bits) or len(bits_out2) != len(pn_bits):
            continue

        ber1 = np.mean(pn_bits != bits_out1)
        ber2 = np.mean(pn_bits != bits_out2)

        # 6. STORAGE: Only append if we got this far (Fixes Index Decoupling)
        csi1 = np.mean(syms_ant1_rotated / ref_syms)
        csi2 = np.mean(syms_ant2_rotated / ref_syms)

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
        collected_syms_ant1.append(syms_ant1_rotated)
        collected_syms_ant2.append(syms_ant2_rotated)

    # Convert lists back to the expected (NumSyms, NumPackets) shape
    if not results:
        return np.array([]), np.array([]), []
        
    return (np.column_stack(collected_syms_ant1), 
            np.column_stack(collected_syms_ant2), 
            results)


def filter_by_ber(results, syms_ant1, syms_ant2, threshold=BER_THRESHOLD):
    keep = [i for i, r in enumerate(results) if r['ber1'] <= threshold and r['ber2'] <= threshold]
    print(f"  BER filter: keeping {len(keep)}/{len(results)} packets (BER ≤ {threshold})")
    return (
        [results[i] for i in keep],
        syms_ant1[:, keep],
        syms_ant2[:, keep],
        len(keep),
    )


def make_results_struct(results):
    return {
        'ber1':     np.array([r['ber1']     for r in results]).reshape(1, -1),
        'cfo':      np.array([r['cfo']      for r in results]).reshape(1, -1),
        'peakVal':  np.array([r['peakVal']  for r in results]).reshape(1, -1),
        'peakTime': np.array([r['peakTime'] for r in results]).reshape(1, -1),
    }


# ==============================================================================
# 3.  BUILD REFERENCE SIGNALS
# ==============================================================================
H_rrc = sio.loadmat("rrcTx_coeffs.mat")["coeffs"].ravel()

ref_syms_S1  = qpsk_modulate(pn_bits_S1)
ref_syms_S2  = qpsk_modulate(pn_bits_S2)
NUM_REF_SYMS = len(ref_syms_S1)

ref_sig_S1 = sio.loadmat('waveform_STTD.mat')['waveform_S1'].ravel()
ref_sig_S2 = sio.loadmat('waveform_STTD.mat')['waveform_S2'].ravel()

GROUP_DELAY = (SPAN * SPS) // 2
ref_sig_S1  = ref_sig_S1[GROUP_DELAY:]
ref_sig_S2  = ref_sig_S2[GROUP_DELAY:]
N_PKT       = len(ref_sig_S1)

# ==============================================================================
# 4.  LOAD IQ DATA
# ==============================================================================
print(f"Loading {args.input} ...")
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
# 5.  PACKET DETECTION
# ==============================================================================
print("Cross-correlation for packet detection...")

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

# ==============================================================================
# 6.  PER-PACKET PROCESSING
# ==============================================================================
print("Processing S1 packets...")
all_syms_ant1_S1, all_syms_ant2_S1, results_S1 = process_packets(valid_packets_S1, rxBuffer, ref_syms_S1, pn_bits_S1)

print("Processing S2 packets...")
all_syms_ant1_S2, all_syms_ant2_S2, results_S2 = process_packets(valid_packets_S2, rxBuffer, ref_syms_S2, pn_bits_S2)

# ==============================================================================
# 7.  BER FILTER
# ==============================================================================
print("\nApplying BER filter...")
results_S1_f, all_syms_ant1_S1_f, all_syms_ant2_S1_f, num_pkts_S1 = filter_by_ber(
    results_S1, all_syms_ant1_S1, all_syms_ant2_S1)
results_S2_f, all_syms_ant1_S2_f, all_syms_ant2_S2_f, num_pkts_S2 = filter_by_ber(
    results_S2, all_syms_ant1_S2, all_syms_ant2_S2)

# ==============================================================================
# 8.  SAVE RESULTS
# ==============================================================================
final_data = {
    'results_S1':             np.array([make_results_struct(results_S1_f)]),
    'results_S2':             np.array([make_results_struct(results_S2_f)]),
    'num_packets_S1':         np.array([[len(results_S1_f)]], dtype=float),
    'num_packets_S2':         np.array([[len(results_S2_f)]], dtype=float),
    'all_syms_ant1_S1':       all_syms_ant1_S1_f,
    'all_syms_ant2_S1':       all_syms_ant2_S1_f,
    'all_syms_ant1_S2':       all_syms_ant1_S2_f,
    'all_syms_ant2_S2':       all_syms_ant2_S2_f,
    'rx_pos':                 rx_pos.reshape(1, -1),
    'tx_pos':                 tx_pos.reshape(1, -1),
    'freq':                   np.array([[915e6]]),
    'tx_bf_angle':            np.array([[np.nan]]),
    'rx_orientation':         np.array([[np.nan]]),
    'tx_mimo':                np.array([[np.nan]]),
}

sio.savemat(args.output, {'final_data': final_data})
print(f"Results saved → {args.output}")

# ==============================================================================
# 9.  (OPTIONAL) PLOT CONSTELLATIONS
# ==============================================================================

# import matplotlib.pyplot as plt

# def plot_constellations(all_syms_ant1, all_syms_ant2, valid_packets, label):
#     sample_idxs = range(min(valid_packets, 10))
#     n_cols = max(len(sample_idxs), 2)
#     fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))
#     if n_cols == 1:
#         axes = axes.reshape(2, 1)
#     for si, p in enumerate(sample_idxs):
#         syms1 = all_syms_ant1[:, p]
#         syms2 = all_syms_ant2[:, p]
#         ax = axes[0, si]
#         ax.scatter(syms1.real, syms1.imag, s=8, c='b', alpha=0.3)
#         ax.grid(True); ax.set_aspect('equal')
#         ax.set_title(f'{label} Pkt {p+1} | Ant1',  fontsize=8)
#         ax.set_xlabel('I'); ax.set_ylabel('Q')
#         ax = axes[1, si]
#         ax.scatter(syms2.real, syms2.imag, s=8, c='r', alpha=0.3)
#         ax.grid(True); ax.set_aspect('equal')
#         ax.set_title(f'{label} Pkt {p+1} | Ant2', fontsize=8)
#         ax.set_xlabel('I'); ax.set_ylabel('Q')
#     for si in range(len(sample_idxs), n_cols):
#         axes[0, si].set_visible(False)
#         axes[1, si].set_visible(False)
#     fig.tight_layout()

# plot_constellations(all_syms_ant1_S1_f, all_syms_ant2_S1_f, num_pkts_S1, "S1")
# plot_constellations(all_syms_ant1_S2_f, all_syms_ant2_S2_f, num_pkts_S2, "S2")
# plt.show()
