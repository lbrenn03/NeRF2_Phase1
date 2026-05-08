# Data Collection Scripts for NeRF2 Pilot 1

This directory contains MATLAB scripts used during the first round of NeRF2 data collection (Pilot 1). These scripts were designed to handle signal transmission and reception using the Software-Defined Radio (SDR) platform for training data acquisition.

## Overview of Scripts

### 1. `data_capture.m`
- **Purpose**: Captures raw IQ data from the receiver (RX) end at a specific position, orientation, and center frequency.
- **Key Parameters**:
  - **Center Frequency**: `915e6` Hz
  - **RX Gain**: `[40, 40]`
  - **Sample Rate (Fs)**: `500e3` samples per second
  - **Capture Time**: `2` seconds
- **Data Saved**:
  - Complex IQ data buffered during the capture session.
  - Metadata: RX (`[x, y, z]`) and TX (`[x, y, z]`) positions, orientation, center frequency, and gain.
  - Saved as `.mat` files in either `pilot1_data/` or `pilot1_test/` based on the RX position.
- **Features**:
  - Ensures dynamic filename generation to avoid overwrites.
  - Error handling to address RX position mismatches or incomplete data captures.

---

### 2. `tx_pilot1.m`
- **Purpose**: Sets up and transmits a beacon signal from the transmitter (TX) for position data collection.
- **Key Parameters**:
  - **Center Frequency**: `915e6` Hz
  - **TX Gain**: `60`
  - **Signal Sampling Rate (Fs)**: `30e6 / 60 = 500e3` samples per second
- **Signal Processing**:
  - **PN Sequence Generation**:
    - Polynomial: `[10 3 0]` (`x^10 + x^3 + 1`)
    - 1023 samples per frame.
  - **QPSK Modulation**:
    - Modulates PN sequence with a $\pi/4$ phase offset.
  - **Raised Cosine Pulse Shaping**:
    - Roll-off factor: `0.25`
    - 5 samples per symbol.
  - **Amplitude Normalization**:
    - Scaled to a maximum amplitude of `0.8` to prevent DAC clipping.
- **Frame Construction & Transmission**:
  - Appends silence (20,000 samples).
  - Continuously transmits the constructed sounding signal frame until stopped (e.g., `Ctrl+C`).

---

## General Notes
- These scripts were used exclusively during the Pilot 1 data collection.
- Ensure SDR hardware parameters (e.g., gain, center frequency, sample rate) are consistent across TX and RX devices during data collection to avoid mismatch errors.
- Each `.mat` file containing raw IQ data is uniquely named based on RX/TX positions and orientations to prevent data overwriting.

---

This directory was designed with modularity to adapt to changing data collection requirements for future pilots.
