"""
MIMO Signal Processor - Interleaved Stream Demodulation
Processes simultaneously transmitted S1 (odd samples) and S2 (even samples)
from TX Ant1 and TX Ant2, received on RX Ant1 and RX Ant2.
"""

import numpy as np
import scipy.io as sio
from scipy import signal
from scipy.cluster.vq import kmeans2
import matplotlib.pyplot as plt

# ==============================================================================
# 1. CONFIGURATION & LOADING
# ==============================================================================
Fs = 500e3
SPS = 5
SPAN = 6
GROUP_DELAY = (SPAN * SPS) // 2  # 15 samples

# Load raw capture
data_in = sio.loadmat('interleaved_mimo2.mat')
raw_iq = data_in['raw_data']['iq'][0, 0]  # (N, 2)
rx_pos = data_in['raw_data']['rx_pos'][0, 0].flatten()
tx_pos = data_in['raw_data']['tx_pos'][0, 0].flatten()

# Load support files
H_rrc = sio.loadmat('rrcTx_coeffs.mat')['coeffs'].ravel()
ref_sig_S1_base = sio.loadmat('reference_sig.mat')['reference_sig'].ravel()

# Load MIMO reference (S2 waveform)
try:
    ref_sig_S2_base = sio.loadmat('reference_sig_mimo.mat')['waveform_S2'].ravel()
except FileNotFoundError:
    print("⚠️  Warning: reference_sig_mimo.mat not found.")
    ref_sig_S2_base = ref_sig_S1_base.copy()

# === CREATE INTERLEAVED REFERENCE SIGNALS (matching transmitter) ===
# TX does: S1 at odd indices (Python 0::2), S2 at even indices (Python 1::2)
# But at the RX antenna, we receive a SUPERPOSITION from both TX antennas

# Create interleaved waveforms as transmitted
waveform_S1_interleaved = np.zeros(2 * len(ref_sig_S1_base), dtype=complex)
waveform_S1_interleaved[0::2] = ref_sig_S1_base

waveform_S2_interleaved = np.zeros(2 * len(ref_sig_S2_base), dtype=complex)
waveform_S2_interleaved[1::2] = ref_sig_S2_base

# The received signal at each antenna is a combination:
# RX Ant1 ≈ H11*S1_interleaved + H12*S2_interleaved
# RX Ant2 ≈ H21*S1_interleaved + H22*S2_interleaved
# For coarse timing sync, we create a composite reference by simply adding them
# (ignoring channel coefficients for now - we're just looking for the pattern)
ref_composite = waveform_S1_interleaved + waveform_S2_interleaved

print(f"Base reference length: {len(ref_sig_S1_base)} samples")
print(f"Interleaved reference length: {len(waveform_S1_interleaved)} samples")
print(f"Composite reference length: {len(ref_composite)} samples")

# DC Removal
rxBuffer = raw_iq - np.mean(raw_iq, axis=0)

print(f"Loaded {rxBuffer.shape[0]} samples from both RX antennas")

# ==============================================================================
# 2. MODULATION UTILITIES
# ==============================================================================
def qpsk_modulate(bits):
    """Gray-coded QPSK with pi/4 offset matching MATLAB."""
    bits = np.asarray(bits).reshape(-1, 2)
    GRAY_MAP = {
        (0, 0): 0, (0, 1): 1, (1, 1): 2, (1, 0): 3
    }
    gray_idx = np.array([GRAY_MAP[tuple(b)] for b in bits], dtype=int)
    return np.exp(1j * (np.pi/4 + np.pi/2 * gray_idx))

def qpsk_demodulate(symbols):
    """Demodulate QPSK symbols to bits (Gray mapping)."""
    candidates = np.exp(1j * (np.pi/4 + np.pi/2 * np.arange(4)))
    dists = np.abs(symbols[:, None] - candidates[None, :])
    idx = np.argmin(dists, axis=1)
    gray_to_bits = np.array([[0,0], [0,1], [1,1], [1,0]], dtype=np.uint8)
    return gray_to_bits[idx].ravel()

# Load reference PN sequences (replace with actual sequences)
# For demonstration, using placeholder - REPLACE with actual pn_bits_S1/S2
pn_bits_S1 = np.asarray([1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,1,1,1,1,1,1,0,1,1,1,0,0,0,1,0,0,1,1,1,1,1,0,0,0,1,1,0,0,1,1,1,1,1,0,1,0,1,1,0,0,1,0,1,1,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,1,1,0,0,1,0,0,1,1,0,1,0,0,0,0,1,0,0,1,0,1,0,1,0,0,0,0,1,1,1,1,0,1,0,1,1,1,0,1,0,1,1,0,1,1,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,1,1,0,0,1,1,0,0,0,0,1,0,1,0,1,1,0,1,0,1,1,1,0,0,0,1,1,0,1,1,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,0,0,1,1,1,1,0,1,1,0,1,1,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,1,0,1,1,0,1,0,1,0,1,0,0,0,1,1,1,1,1,0,1,1,1,1,0,0,1,0,0,1,0,1,1,0,0,0,0,0,1,0,0,1,1,0,0,1,0,0,0,1,0,1,0,0,0,1,1,0,1,1,0,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,1,1,1,0,1,1,1,1,1,1,1,0,0,1,0,0,0,0,1,1,0,0,0,1,0,1,1,0,1,1,1,0,1,0,0,0,0,1,1,0,1,0,1,0,1,1,0,0,1,1,1,1,0,0,1,0,1,1,0,1,1,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,1,1,0,0,0,0,0,0,1,0,1,1,0,0,0,1,0,1,0,0,1,1,1,0,1,1,0,0,1,1,1,0,0,0,1,0,1,1,1,1,1,1,0,1,0,1,0,0,0,1,0,1,1,1,0,1,1,0,1,0,1,1,0,0,0,0,1,1,0,0,1,1,0,1,1,0,1,0,1,0,0,0,0,0,1,1,1,0,1,0,0,1,1,1,1,0,1,0,0,1,1,0,1,0,1,0,0,1,0,0,1,1,1,0,0,0,0,0,1,1,1,1,1,0,0,1,1,1,0,0,1,1,0,1,1,1,1,0,1,0,0,0,1,0,1,0,1,0,1,1,0,1,1,1,1,1,0,0,0,0,1,0,0,1,1,1,0,1,0,0,0,1,1,1,0,1,0,1,1,1,1,1,0,1,1,0,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,1,0,0,1,0,1,0,1,1,0,0,0,1,1,1,0,0,1,1,1,1,1,1,1,0,1,1,0,0,0,0,1,0,0,0,1,1,0,1,0,0,1,1,1,0,0,1,0,0,1,1,1,1,0,0,0,0,1,1,0,1,1,1,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,0,1,1,1,1,1,0,1,0,0,1,0,0,1,0,1,0,0,0,0,0,0,1,1,0,1,0,0,0,1,1,0,0,1,0,1,1,1,0,1,0,0,1,0,1,1,0,1,0,0,0,1,0,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1,0,1,0,0,1,0,0,0,1,1,0,0,0,0,1,1,1,0,1,1,0,1,1,1,1,0,0,0,0,0,1,0,1,1,1,0,0,1,0,1,0,1,1,1,0,0,1,1,1,0,1,1,1,0,1,1,1,0,0,1,1,0,0,1,1,1,0,1,0,1,0,1,1,1,0,1,1,1,1,0,1,1,0,0,1,0,1,0,0,0,1,0,0,1,1,0,1,1,0,0,0,1,0,0,0,0,1,1,1,0,0,1,0,1,1,1,1,1,0,0,1,0,1,0,0,1,1,0,0,1,1,0,0,1,0,1,0,1,0,1,0,0,1,1,1,1,1,1,0,0,1,1,0,0,0,1,1,0,1,0,1,1,1,1,0,0,1,1,0,1,0,1,1,0,1,0,0,1,1,0,0,0,1,0,0,1,0,1,1,1,0,0,0,0,1,0,1,1,1,1,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,0,1,0,0,0,0,0,1,0,1,0,1,0,0,1,0,1,1,1,1,0,0,0,1,0,1,0,1,1,1,1,0,1,1,1,0,1,0,1,0,0,1,1,0,1,1,1,0,0,1,0,0,0,1,1,1,0,0,0,1])
pn_bits_S2 = np.asarray([0,1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,0,1,0,0,0,0,0,1,0,1,0,1,0,0,1,0,1,1,1,1,0,0,0,1,0,1,0,1,1,1,1,0,1,1,1,0,1,0,1,0,0,1,1,0,1,1,1,0,0,1,0,0,0,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,1,1,1,1,1,1,0,1,1,1,0,0,0,1,0,0,1,1,1,1,1,0,0,0,1,1,0,0,1,1,1,1,1,0,1,0,1,1,0,0,1,0,1,1,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,1,1,0,0,1,0,0,1,1,0,1,0,0,0,0,1,0,0,1,0,1,0,1,0,0,0,0,1,1,1,1,0,1,0,1,1,1,0,1,0,1,1,0,1,1,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,1,1,0,0,1,1,0,0,0,0,1,0,1,0,1,1,0,1,0,1,1,1,0,0,0,1,1,0,1,1,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,0,0,1,1,1,1,0,1,1,0,1,1,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,1,0,1,1,0,1,0,1,0,1,0,0,0,1,1,1,1,1,0,1,1,1,1,0,0,1,0,0,1,0,1,1,0,0,0,0,0,1,0,0,1,1,0,0,1,0,0,0,1,0,1,0,0,0,1,1,0,1,1,0,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,1,1,1,0,1,1,1,1,1,1,1,0,0,1,0,0,0,0,1,1,0,0,0,1,0,1,1,0,1,1,1,0,1,0,0,0,0,1,1,0,1,0,1,0,1,1,0,0,1,1,1,1,0,0,1,0,1,1,0,1,1,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,1,1,0,0,0,0,0,0,1,0,1,1,0,0,0,1,0,1,0,0,1,1,1,0,1,1,0,0,1,1,1,0,0,0,1,0,1,1,1,1,1,1,0,1,0,1,0,0,0,1,0,1,1,1,0,1,1,0,1,0,1,1,0,0,0,0,1,1,0,0,1,1,0,1,1,0,1,0,1,0,0,0,0,0,1,1,1,0,1,0,0,1,1,1,1,0,1,0,0,1,1,0,1,0,1,0,0,1,0,0,1,1,1,0,0,0,0,0,1,1,1,1,1,0,0,1,1,1,0,0,1,1,0,1,1,1,1,0,1,0,0,0,1,0,1,0,1,0,1,1,0,1,1,1,1,1,0,0,0,0,1,0,0,1,1,1,0,1,0,0,0,1,1,1,0,1,0,1,1,1,1,1,0,1,1,0,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,1,0,0,1,0,1,0,1,1,0,0,0,1,1,1,0,0,1,1,1,1,1,1,1,0,1,1,0,0,0,0,1,0,0,0,1,1,0,1,0,0,1,1,1,0,0,1,0,0,1,1,1,1,0,0,0,0,1,1,0,1,1,1,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,0,1,1,1,1,1,0,1,0,0,1,0,0,1,0,1,0,0,0,0,0,0,1,1,0,1,0,0,0,1,1,0,0,1,0,1,1,1,0,1,0,0,1,0,1,1,0,1,0,0,0,1,0,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1,0,1,0,0,1,0,0,0,1,1,0,0,0,0,1,1,1,0,1,1,0,1,1,1,1,0,0,0,0,0,1,0,1,1,1,0,0,1,0,1,0,1,1,1,0,0,1,1,1,0,1,1,1,0,1,1,1,0,0,1,1,0,0,1,1,1,0,1,0,1,0,1,1,1,0,1,1,1,1,0,1,1,0,0,1,0,1,0,0,0,1,0,0,1,1,0,1,1,0,0,0,1,0,0,0,0,1,1,1,0,0,1,0,1,1,1,1,1,0,0,1,0,1,0,0,1,1,0,0,1,1,0,0,1,0,1,0,1,0,1,0,0,1,1,1,1,1,1,0,0,1,1,0,0,0,1,1,0,1,0,1,1,1,1,0,0,1,1,0,1,0,1,1,0,1,0,0,1,1,0,0,0,1,0,0,1,0,1,1,1,0,0,0,0,1,0,1,1,1,1,0])

ref_syms_S1 = qpsk_modulate(pn_bits_S1)
ref_syms_S2 = qpsk_modulate(pn_bits_S2)
NUM_REF_SYMS = len(ref_syms_S1)

print(f"Reference symbols: {NUM_REF_SYMS} per stream")

# ==============================================================================
# 3. PACKET DETECTION (using COMPOSITE interleaved reference)
# ==============================================================================
def find_packets(sig, ref, prominence_frac=0.6, min_spacing_sec=0.05):
    """Find packet start indices via cross-correlation."""
    xcorr = signal.correlate(sig, ref, mode='valid')
    env = np.abs(xcorr)**2
    min_dist = int(min_spacing_sec * Fs * 2)  # *2 because interleaved is 2x longer
    peaks, props = signal.find_peaks(
        env, 
        prominence=prominence_frac * np.max(env), 
        distance=min_dist
    )
    return peaks, env[peaks]

# Detect using COMPOSITE reference (S1 + S2 interleaved together)
# CRITICAL: Process each antenna independently due to different spatial locations
print("Performing coarse timing sync with composite interleaved reference...")
print("Processing RX Antenna 1...")
peaks_ant1, vals_ant1 = find_packets(rxBuffer[:, 0], ref_composite)

print("Processing RX Antenna 2...")
peaks_ant2, vals_ant2 = find_packets(rxBuffer[:, 1], ref_composite)

# Sort by peak strength and take top N packets per antenna
NUM_PACKETS_EXPECTED = 10  # Adjust based on your capture
peaks_ant1 = peaks_ant1[np.argsort(vals_ant1)[::-1][:NUM_PACKETS_EXPECTED]]
peaks_ant2 = peaks_ant2[np.argsort(vals_ant2)[::-1][:NUM_PACKETS_EXPECTED]]

# Sort by time
peaks_ant1 = np.sort(peaks_ant1)
peaks_ant2 = np.sort(peaks_ant2)

num_p_ant1 = len(peaks_ant1)
num_p_ant2 = len(peaks_ant2)

print(f"✅ Antenna 1: Detected {num_p_ant1} packets")
print(f"✅ Antenna 2: Detected {num_p_ant2} packets\n")

# ==============================================================================
# 4. RRC MATCHED FILTER & CFO ESTIMATION
# ==============================================================================
def rrc_rx_filter(x, h, sps, span):
    """Apply RRC matched filter with decimation."""
    y = np.convolve(x, h, mode='full')
    delay = (len(h) - 1) // 2
    y = y[delay:]  # Remove group delay
    return y[::sps]  # Decimate

def coarse_cfo_estimate(x, fs, modulation_order=4):
    """
    4th-power FFT CFO estimation for QPSK.
    Raises signal to 4th power to remove QPSK modulation,
    finds spectral peak, divides by 4 to recover CFO.
    """
    x4 = x ** modulation_order
    nfft = 1 << (len(x4)-1).bit_length()  # Next power of 2
    X4 = np.fft.fft(x4, n=nfft)
    freqs = np.fft.fftfreq(nfft, d=1/fs)
    
    mag = np.abs(X4)
    mag[0] = 0.0  # Suppress DC
    
    peak_idx = np.argmax(mag)
    cfo_est = freqs[peak_idx] / modulation_order
    
    return cfo_est

def fine_cfo_estimate(syms, ref_syms):
    """
    Fine CFO estimation using pilot symbols.
    Compares received symbols to reference and extracts phase slope.
    """
    L = min(len(syms), len(ref_syms))
    
    # Compute phase difference between received and reference
    phase_diff = np.angle(syms[:L] * np.conj(ref_syms[:L]))
    
    # Unwrap phase to handle 2π discontinuities
    phase_unwrapped = np.unwrap(phase_diff)
    
    # Linear fit to get phase slope (CFO causes linear phase drift)
    indices = np.arange(L)
    coeffs = np.polyfit(indices, phase_unwrapped, 1)
    
    # Slope in radians per symbol
    phase_slope = coeffs[0]
    
    return phase_slope

# ==============================================================================
# 5. MIMO PROCESSING LOOP (Per-Antenna Processing)
# ==============================================================================
# Process Antenna 1 packets
all_syms_S1_from_ant1 = np.zeros((NUM_REF_SYMS, num_p_ant1), dtype=complex)
all_syms_S2_from_ant1 = np.zeros((NUM_REF_SYMS, num_p_ant1), dtype=complex)

# Process Antenna 2 packets
all_syms_S1_from_ant2 = np.zeros((NUM_REF_SYMS, num_p_ant2), dtype=complex)
all_syms_S2_from_ant2 = np.zeros((NUM_REF_SYMS, num_p_ant2), dtype=complex)

H_matrices_ant1 = np.zeros((num_p_ant1, 2, 2), dtype=complex)
H_matrices_ant2 = np.zeros((num_p_ant2, 2, 2), dtype=complex)

results_ant1 = []
results_ant2 = []

# ============================================================================
# PROCESS ANTENNA 1 PACKETS
# ============================================================================
print("=" * 60)
print("PROCESSING RX ANTENNA 1 PACKETS")
print("=" * 60)

for p in range(num_p_ant1):
    peak_idx = peaks_ant1[p]
    # Extract enough samples for interleaved processing
    pkt_len_samples = len(ref_composite) + 200
    end_idx = min(peak_idx + pkt_len_samples, rxBuffer.shape[0])
    
    # IMPORTANT: Only extract from Antenna 1 for this packet
    seg = rxBuffer[peak_idx:end_idx, 0]
    
    # --- DEINTERLEAVE ---
    best_nmse = np.inf
    final_mf1, final_mf2 = None, None
    best_offset = 0
    
    for offset in [0, 1]:
        # Deinterleave to separate S1 and S2 from the composite signal
        s1_samples = seg[offset::2]
        s2_samples = seg[(1-offset)::2]
        
        # Apply matched filter
        mf1 = rrc_rx_filter(s1_samples, H_rrc, SPS, SPAN)
        mf2 = rrc_rx_filter(s2_samples, H_rrc, SPS, SPAN)
        
        # Validate against reference S1
        L_test = min(len(mf1), NUM_REF_SYMS)
        if L_test < 10:
            continue
            
        phase_ref = np.angle(np.mean(mf1[:L_test] / ref_syms_S1[:L_test]))
        test_syms = mf1[:L_test] * np.exp(-1j * phase_ref)
        
        nmse = np.mean(np.abs(test_syms - ref_syms_S1[:L_test])**2) / \
               np.mean(np.abs(ref_syms_S1[:L_test])**2)
        
        if nmse < best_nmse:
            best_nmse = nmse
            best_offset = offset
            final_mf1, final_mf2 = mf1, mf2
    
    print(f"  Pkt {p}: offset={best_offset}, NMSE={10*np.log10(best_nmse):.1f} dB")
    
    # Phase anchoring
    L = min(len(final_mf1), NUM_REF_SYMS)
    anch1 = np.angle(np.mean(final_mf1[:L] / ref_syms_S1[:L]))
    anch2 = np.angle(np.mean(final_mf2[:L] / ref_syms_S2[:L]))
    
    final_mf1 *= np.exp(-1j * anch1)
    final_mf2 *= np.exp(-1j * anch2)
    
    # Storage
    all_syms_S1_from_ant1[:, p] = final_mf1[:NUM_REF_SYMS]
    all_syms_S2_from_ant1[:, p] = final_mf2[:NUM_REF_SYMS]
    
    # MIMO Channel Estimation for Antenna 1
    N_est = min(100, L)
    R_matrix = np.column_stack([
        all_syms_S1_from_ant1[:N_est, p], 
        all_syms_S2_from_ant1[:N_est, p]
    ])
    S_matrix = np.column_stack([
        ref_syms_S1[:N_est], 
        ref_syms_S2[:N_est]
    ])
    
    h_row1, _, _, _ = np.linalg.lstsq(S_matrix, R_matrix[:, 0], rcond=None)
    h_row2, _, _, _ = np.linalg.lstsq(S_matrix, R_matrix[:, 1], rcond=None)
    H_est = np.vstack([h_row1, h_row2])
    H_matrices_ant1[p] = H_est
    
    R_pred = S_matrix @ H_est.T
    error = np.mean(np.abs(R_matrix - R_pred)**2)
    sig_pwr = np.mean(np.abs(R_matrix)**2)
    nmse_db = 10 * np.log10(error / (sig_pwr + 1e-12))
    
    results_ant1.append({
        'nmse_db': nmse_db,
        'H_est': H_est,
        'peakTime': peak_idx / Fs
    })

# ============================================================================
# PROCESS ANTENNA 2 PACKETS
# ============================================================================
print("\n" + "=" * 60)
print("PROCESSING RX ANTENNA 2 PACKETS")
print("=" * 60)

for p in range(num_p_ant2):
    peak_idx = peaks_ant2[p]
    
    pkt_len_samples = len(ref_composite) + 200
    end_idx = min(peak_idx + pkt_len_samples, rxBuffer.shape[0])
    
    # IMPORTANT: Only extract from Antenna 2 for this packet
    seg = rxBuffer[peak_idx:end_idx, 1]
    
    # --- DEINTERLEAVE ---
    best_nmse = np.inf
    final_mf1, final_mf2 = None, None
    best_offset = 0
    
    for offset in [0, 1]:
        s1_samples = seg[offset::2]
        s2_samples = seg[(1-offset)::2]
        
        mf1 = rrc_rx_filter(s1_samples, H_rrc, SPS, SPAN)
        mf2 = rrc_rx_filter(s2_samples, H_rrc, SPS, SPAN)
        
        L_test = min(len(mf1), NUM_REF_SYMS)
        if L_test < 10:
            continue
            
        phase_ref = np.angle(np.mean(mf1[:L_test] / ref_syms_S1[:L_test]))
        test_syms = mf1[:L_test] * np.exp(-1j * phase_ref)
        
        nmse = np.mean(np.abs(test_syms - ref_syms_S1[:L_test])**2) / \
               np.mean(np.abs(ref_syms_S1[:L_test])**2)
        
        if nmse < best_nmse:
            best_nmse = nmse
            best_offset = offset
            final_mf1, final_mf2 = mf1, mf2
    
    print(f"  Pkt {p}: offset={best_offset}, NMSE={10*np.log10(best_nmse):.1f} dB")
    
    # Phase anchoring
    L = min(len(final_mf1), NUM_REF_SYMS)
    anch1 = np.angle(np.mean(final_mf1[:L] / ref_syms_S1[:L]))
    anch2 = np.angle(np.mean(final_mf2[:L] / ref_syms_S2[:L]))
    
    final_mf1 *= np.exp(-1j * anch1)
    final_mf2 *= np.exp(-1j * anch2)
    
    # Storage
    all_syms_S1_from_ant2[:, p] = final_mf1[:NUM_REF_SYMS]
    all_syms_S2_from_ant2[:, p] = final_mf2[:NUM_REF_SYMS]
    
    # MIMO Channel Estimation for Antenna 2
    N_est = min(100, L)
    R_matrix = np.column_stack([
        all_syms_S1_from_ant2[:N_est, p], 
        all_syms_S2_from_ant2[:N_est, p]
    ])
    S_matrix = np.column_stack([
        ref_syms_S1[:N_est], 
        ref_syms_S2[:N_est]
    ])
    
    h_row1, _, _, _ = np.linalg.lstsq(S_matrix, R_matrix[:, 0], rcond=None)
    h_row2, _, _, _ = np.linalg.lstsq(S_matrix, R_matrix[:, 1], rcond=None)
    H_est = np.vstack([h_row1, h_row2])
    H_matrices_ant2[p] = H_est
    
    R_pred = S_matrix @ H_est.T
    error = np.mean(np.abs(R_matrix - R_pred)**2)
    sig_pwr = np.mean(np.abs(R_matrix)**2)
    nmse_db = 10 * np.log10(error / (sig_pwr + 1e-12))
    
    results_ant2.append({
        'nmse_db': nmse_db,
        'H_est': H_est,
        'peakTime': peak_idx / Fs
    })

print("\n" + "=" * 60)

# ==============================================================================
# 6. BER COMPUTATION (Per-Antenna)
# ==============================================================================
print("\n" + "=" * 60)
print("BER ANALYSIS - RX ANTENNA 1")
print("=" * 60)

total_bits = NUM_REF_SYMS * 2
total_err_s1_ant1 = 0
total_err_s2_ant1 = 0

print(f"{'Packet':<8} | {'BER S1':<12} | {'BER S2':<12}")
print("-" * 40)

for p in range(num_p_ant1):
    bits_rx_s1 = qpsk_demodulate(all_syms_S1_from_ant1[:, p])
    bits_rx_s2 = qpsk_demodulate(all_syms_S2_from_ant1[:, p])
    
    err_s1 = np.sum(bits_rx_s1 != pn_bits_S1[:total_bits])
    err_s2 = np.sum(bits_rx_s2 != pn_bits_S2[:total_bits])
    
    total_err_s1_ant1 += err_s1
    total_err_s2_ant1 += err_s2
    
    ber_s1 = err_s1 / total_bits
    ber_s2 = err_s2 / total_bits
    print(f"{p:<8} | {ber_s1:<12.5f} | {ber_s2:<12.5f}")

overall_ber_s1_ant1 = total_err_s1_ant1 / (num_p_ant1 * total_bits)
overall_ber_s2_ant1 = total_err_s2_ant1 / (num_p_ant1 * total_bits)

print("-" * 40)
print(f"OVERALL  | {overall_ber_s1_ant1:<12.5f} | {overall_ber_s2_ant1:<12.5f}\n")

print("=" * 60)
print("BER ANALYSIS - RX ANTENNA 2")
print("=" * 60)

total_err_s1_ant2 = 0
total_err_s2_ant2 = 0

print(f"{'Packet':<8} | {'BER S1':<12} | {'BER S2':<12}")
print("-" * 40)

for p in range(num_p_ant2):
    bits_rx_s1 = qpsk_demodulate(all_syms_S1_from_ant2[:, p])
    bits_rx_s2 = qpsk_demodulate(all_syms_S2_from_ant2[:, p])
    
    err_s1 = np.sum(bits_rx_s1 != pn_bits_S1[:total_bits])
    err_s2 = np.sum(bits_rx_s2 != pn_bits_S2[:total_bits])
    
    total_err_s1_ant2 += err_s1
    total_err_s2_ant2 += err_s2
    
    ber_s1 = err_s1 / total_bits
    ber_s2 = err_s2 / total_bits
    print(f"{p:<8} | {ber_s1:<12.5f} | {ber_s2:<12.5f}")

overall_ber_s2_ant2 = total_err_s1_ant2 / (num_p_ant2 * total_bits)
overall_ber_s2_ant2 = total_err_s2_ant2 / (num_p_ant2 * total_bits)

print("-" * 40)
print(f"OVERALL  | {overall_ber_s2_ant2:<12.5f} | {overall_ber_s2_ant2:<12.5f}\n")

# ==============================================================================
# 7. VISUALIZATION
# ==============================================================================

# --- A. Correlation Peaks (Both Antennas) ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

# Antenna 1
xcorr_ant1 = signal.correlate(rxBuffer[:, 0], ref_composite, mode='valid')
time_axis = np.arange(len(xcorr_ant1)) / Fs
ax1.plot(time_axis, np.abs(xcorr_ant1)**2, label='RX Ant 1 Xcorr', alpha=0.7, color='blue')
for pk in peaks_ant1:
    if pk < len(xcorr_ant1):
        ax1.axvline(pk/Fs, color='red', linestyle='--', alpha=0.5)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('|Xcorr|²')
ax1.set_title('Packet Detection - RX Antenna 1')
ax1.legend()
ax1.grid(True)

# Antenna 2
xcorr_ant2 = signal.correlate(rxBuffer[:, 1], ref_composite, mode='valid')
ax2.plot(time_axis, np.abs(xcorr_ant2)**2, label='RX Ant 2 Xcorr', alpha=0.7, color='green')
for pk in peaks_ant2:
    if pk < len(xcorr_ant2):
        ax2.axvline(pk/Fs, color='red', linestyle='--', alpha=0.5)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('|Xcorr|²')
ax2.set_title('Packet Detection - RX Antenna 2')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# --- B. Constellation Plots ---
pkts_to_show = [0, min(5, num_p_ant1-1), min(10, num_p_ant1-1)]
fig, axs = plt.subplots(4, 3, figsize=(15, 16))

for i, p_idx in enumerate(pkts_to_show):
    if p_idx < num_p_ant1:
        # RX Ant1 - Stream S1
        axs[0, i].scatter(all_syms_S1_from_ant1[:, p_idx].real, 
                          all_syms_S1_from_ant1[:, p_idx].imag, 
                          s=8, alpha=0.5, c='blue')
        axs[0, i].set_title(f'RX Ant1 Pkt{p_idx} - S1')
        axs[0, i].set_xlim([-1.5, 1.5])
        axs[0, i].set_ylim([-1.5, 1.5])
        axs[0, i].grid(True, alpha=0.3)
        axs[0, i].axhline(0, color='k', linewidth=0.5)
        axs[0, i].axvline(0, color='k', linewidth=0.5)
        
        # RX Ant1 - Stream S2
        axs[1, i].scatter(all_syms_S2_from_ant1[:, p_idx].real, 
                          all_syms_S2_from_ant1[:, p_idx].imag, 
                          s=8, alpha=0.5, c='orange')
        axs[1, i].set_title(f'RX Ant1 Pkt{p_idx} - S2')
        axs[1, i].set_xlim([-1.5, 1.5])
        axs[1, i].set_ylim([-1.5, 1.5])
        axs[1, i].grid(True, alpha=0.3)
        axs[1, i].axhline(0, color='k', linewidth=0.5)
        axs[1, i].axvline(0, color='k', linewidth=0.5)
    
    if p_idx < num_p_ant2:
        # RX Ant2 - Stream S1
        axs[2, i].scatter(all_syms_S1_from_ant2[:, p_idx].real, 
                          all_syms_S1_from_ant2[:, p_idx].imag, 
                          s=8, alpha=0.5, c='blue')
        axs[2, i].set_title(f'RX Ant2 Pkt{p_idx} - S1')
        axs[2, i].set_xlim([-1.5, 1.5])
        axs[2, i].set_ylim([-1.5, 1.5])
        axs[2, i].grid(True, alpha=0.3)
        axs[2, i].axhline(0, color='k', linewidth=0.5)
        axs[2, i].axvline(0, color='k', linewidth=0.5)
        
        # RX Ant2 - Stream S2
        axs[3, i].scatter(all_syms_S2_from_ant2[:, p_idx].real, 
                          all_syms_S2_from_ant2[:, p_idx].imag, 
                          s=8, alpha=0.5, c='orange')
        axs[3, i].set_title(f'RX Ant2 Pkt{p_idx} - S2')
        axs[3, i].set_xlim([-1.5, 1.5])
        axs[3, i].set_ylim([-1.5, 1.5])
        axs[3, i].grid(True, alpha=0.3)
        axs[3, i].axhline(0, color='k', linewidth=0.5)
        axs[3, i].axvline(0, color='k', linewidth=0.5)

plt.tight_layout()
plt.show()

# ==============================================================================
# 8. K-MEANS CLUSTERING & DATA PACKAGING
# ==============================================================================
cluster_means_S1_ant1 = np.zeros((4, num_p_ant1), dtype=complex)
cluster_means_S2_ant1 = np.zeros((4, num_p_ant1), dtype=complex)
cluster_means_S1_ant2 = np.zeros((4, num_p_ant2), dtype=complex)
cluster_means_S2_ant2 = np.zeros((4, num_p_ant2), dtype=complex)

# Cluster Antenna 1 packets
for p in range(num_p_ant1):
    pts1 = np.column_stack([all_syms_S1_from_ant1[:, p].real, 
                            all_syms_S1_from_ant1[:, p].imag])
    pts2 = np.column_stack([all_syms_S2_from_ant1[:, p].real, 
                            all_syms_S2_from_ant1[:, p].imag])
    
    cent1, _ = kmeans2(pts1, 4, iter=20, minit='points')
    cent2, _ = kmeans2(pts2, 4, iter=20, minit='points')
    
    cluster_means_S1_ant1[:, p] = cent1[:, 0] + 1j * cent1[:, 1]
    cluster_means_S2_ant1[:, p] = cent2[:, 0] + 1j * cent2[:, 1]

# Cluster Antenna 2 packets
for p in range(num_p_ant2):
    pts1 = np.column_stack([all_syms_S1_from_ant2[:, p].real, 
                            all_syms_S1_from_ant2[:, p].imag])
    pts2 = np.column_stack([all_syms_S2_from_ant2[:, p].real, 
                            all_syms_S2_from_ant2[:, p].imag])
    
    cent1, _ = kmeans2(pts1, 4, iter=20, minit='points')
    cent2, _ = kmeans2(pts2, 4, iter=20, minit='points')
    
    cluster_means_S1_ant2[:, p] = cent1[:, 0] + 1j * cent1[:, 1]
    cluster_means_S2_ant2[:, p] = cent2[:, 0] + 1j * cent2[:, 1]

# Save output
final_data = {
    # Antenna 1 data
    'results_ant1': results_ant1,
    'num_packets_ant1': num_p_ant1,
    'all_syms_S1_from_ant1': all_syms_S1_from_ant1,
    'all_syms_S2_from_ant1': all_syms_S2_from_ant1,
    'cluster_means_S1_ant1': cluster_means_S1_ant1,
    'cluster_means_S2_ant1': cluster_means_S2_ant1,
    'H_matrices_ant1': H_matrices_ant1,
    'H_avg_ant1': np.mean(H_matrices_ant1, axis=0),
    'ber_s1_ant1': overall_ber_s1_ant1,
    'ber_s2_ant1': overall_ber_s2_ant1,
    
    # Antenna 2 data
    'results_ant2': results_ant2,
    'num_packets_ant2': num_p_ant2,
    'all_syms_S1_from_ant2': all_syms_S1_from_ant2,
    'all_syms_S2_from_ant2': all_syms_S2_from_ant2,
    'cluster_means_S1_ant2': cluster_means_S1_ant2,
    'cluster_means_S2_ant2': cluster_means_S2_ant2,
    'H_matrices_ant2': H_matrices_ant2,
    'H_avg_ant2': np.mean(H_matrices_ant2, axis=0),
    'ber_s1_ant2': overall_ber_s2_ant2,
    'ber_s2_ant2': overall_ber_s2_ant2,
    
    # Metadata
    'rx_pos': rx_pos,
    'tx_pos': tx_pos,
}

sio.savemat('final_mimo_processed.mat', {'final_data': final_data})
print("✅ Processing complete. Data saved to final_mimo_processed.mat")
