"""
F2 Receiver — Offline Processing
Loads a pre-captured .mat file and outputs nerf_<tag>.mat matching the MATLAB final_data struct layout.

Usage:
    python mimo_proc.py --input raw_capture.mat --output processed.mat
"""

import argparse
import sys
import numpy as np
import scipy.io as sio
import scipy.signal as sps_sig
from scipy.cluster.vq import kmeans2

parser = argparse.ArgumentParser(description='NeRF2 offline receiver')
parser.add_argument('--input',  default='mimo_tx3_11.5_9.5_45.mat', help='Input .mat file')
parser.add_argument('--output', default='processed.mat',             help='Output .mat file')
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


def kmeans_cluster_means(syms):
    pts = np.column_stack([syms.real, syms.imag])
    centroids, _ = kmeans2(pts, 4, iter=20, minit='points', missing='warn')
    return centroids[:, 0] + 1j * centroids[:, 1]


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
    num_p = len(valid_packets)
    all_syms_ant1 = np.zeros((NUM_REF_SYMS, num_p), dtype=complex)
    all_syms_ant2 = np.zeros((NUM_REF_SYMS, num_p), dtype=complex)
    results = []

    for p, pkt in enumerate(valid_packets):
        s = pkt['startIdx'] - 1000
        extraction_len = N_PKT - (SPAN // 2 * SPS) + 2000

        pkt_ant1 = rxBuffer[s:s + extraction_len, 0]
        pkt_ant2 = rxBuffer[s:s + extraction_len, 1]

        if len(pkt_ant1) != extraction_len or len(pkt_ant2) != extraction_len:
            continue

        est_cfo = coarse_cfo_estimate(pkt_ant1, Fs)
        t_vec = np.arange(extraction_len) / Fs
        cfo_vec = np.exp(-1j * 2 * np.pi * est_cfo * t_vec)

        clean_ant1 = pkt_ant1 * cfo_vec
        clean_ant2 = pkt_ant2 * cfo_vec

        mf_ant1 = rrc_rx_filter(clean_ant1, H_rrc, SPS, ref_syms=ref_syms)
        mf_ant2 = rrc_rx_filter(clean_ant2, H_rrc, SPS, ref_syms=ref_syms)

        fine_corr1, fine_lags1 = xcorr(mf_ant1, ref_syms)
        fine_idx1 = int(np.argmax(np.abs(fine_corr1)))
        los_phase_ref = np.angle(fine_corr1[fine_idx1])
        phase_anchor = np.exp(-1j * los_phase_ref)

        fine_corr2, fine_lags2 = xcorr(mf_ant2, ref_syms)
        fine_idx2 = int(np.argmax(np.abs(fine_corr2)))

        mf_ant1_anchored = mf_ant1 * phase_anchor
        mf_ant2_anchored = mf_ant2 * phase_anchor

        fine_timing_offset1 = int(fine_lags1[fine_idx1])
        start_idx1 = max(0, min(fine_timing_offset1, len(mf_ant1) - NUM_REF_SYMS))
        end_idx1 = start_idx1 + NUM_REF_SYMS

        fine_timing_offset2 = int(fine_lags2[fine_idx2])
        start_idx2 = max(0, min(fine_timing_offset2, len(mf_ant2) - NUM_REF_SYMS))
        end_idx2 = start_idx2 + NUM_REF_SYMS

        syms_ant1 = mf_ant1_anchored[start_idx1:end_idx1]
        syms_ant2 = mf_ant2_anchored[start_idx2:end_idx2]

        all_syms_ant1[:, p] = fit_to(syms_ant1, NUM_REF_SYMS)
        all_syms_ant2[:, p] = fit_to(syms_ant2, NUM_REF_SYMS)

        syms_to_demod1 = mf_ant1[start_idx1:end_idx1]
        syms_to_demod2 = mf_ant2[start_idx2:end_idx2]

        num_ref = min(len(syms_to_demod1), len(ref_syms))
        phase_corr1 = np.angle(np.sum(syms_to_demod1[:num_ref] * np.conj(ref_syms[:num_ref])))
        phase_corr2 = np.angle(np.sum(syms_to_demod2[:num_ref] * np.conj(ref_syms[:num_ref])))

        bits_out1 = qpsk_demodulate(syms_to_demod1 * np.exp(-1j * phase_corr1))
        bits_out2 = qpsk_demodulate(syms_to_demod2 * np.exp(-1j * phase_corr2))

        L = min(len(pn_bits), len(bits_out1))
        ber1 = int(np.sum(pn_bits[:L] != bits_out1[:L])) / L
        ber2 = int(np.sum(pn_bits[:L] != bits_out2[:L])) / L

        csi1 = np.mean(syms_ant1 / ref_syms[:len(syms_ant1)])
        csi2 = np.mean(syms_ant2 / ref_syms[:len(syms_ant2)])

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
# 8.  CLUSTERING
# ==============================================================================
print("Computing cluster means...")
cluster_means_ant1_S1_f, cluster_means_ant2_S1_f = cluster_packets(all_syms_ant1_S1_f, all_syms_ant2_S1_f)
cluster_means_ant1_S2_f, cluster_means_ant2_S2_f = cluster_packets(all_syms_ant1_S2_f, all_syms_ant2_S2_f)

# ==============================================================================
# 9.  SAVE RESULTS
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
