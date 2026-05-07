# Data Collection Scripts for NeRF2 Training

This directory contains MATLAB scripts used for data collection and signal processing to train the NeRF2 system. Each script is tailored for a specific role in the data acquisition pipeline, with a focus on transmitting and receiving signals using a Software-Defined Radio (SDR) platform.

## Overview of Scripts
### 1. `data_capture.m`
- **Purpose**: Captures raw IQ data for training, using a specified receiver (RX) position and orientation.
- **Key Parameters**:
  - **Center Frequency**: `915e6` Hz
  - **Gain**: `40`
  - **Sample Rate (Fs)**: `500e3` samples per second
  - **Capture Time**: `2` seconds
- **Data Saved**:
  - Metadata about the capture (RX position, TX position, frequency, etc.)
  - IQ samples saved to a `.mat` file (`raw_data`) with descriptive filenames.
- **Special Features**:
  - Dynamically adjusts filenames based on RX and TX positions to prevent overwrites.
  - Includes error handling for mismatched positions or duplicate filenames.

---

### 2. `data_capture_f2.m`
- **Purpose**: Similar to `data_capture.m` but tailored for a higher frequency band.
- **Key Parameters**:
  - **Center Frequency**: `5.4e9` Hz
  - **Gain**: `70`
  - **Sample Rate (Fs)**: `500e3` samples per second
  - **Capture Time**: `2` seconds
- **Planned Features**:
  - All other functionalities are identical to `data_capture.m`.

---

### 3. `f2_tx_mimo.m`
- **Purpose**: Implements a MIMO transmitter configuration for high-frequency bands, specifically for `5.4e9` Hz.
- **Key Parameters**:
  - **Center Frequency**: `5.4e9` Hz
  - **Gain**: `[80, 80]` for the two MIMO channels
  - **Sample Rate (Fs)**: `30e6 / 60 = 500e3` samples per second
- **Key Features**:
  - Generates pseudo-random noise (PN) sequences as sounding bits.
  - Modulates using QPSK, applies Raised Cosine Pulse Shaping, and normalizes waveforms.
  - Continually alternates between two MIMO streams during transmission.

---

### 4. `tx_mimo.m`
- **Purpose**: Implements a MIMO transmitter for low-frequency bands, specifically for `915e6` Hz.
- **Key Parameters**:
  - **Center Frequency**: `915e6` Hz
  - **Gain**: `[60, 60]` for the two MIMO channels
  - **Sample Rate (Fs)**: `30e6 / 60 = 500e3` samples per second
- **Key Features**:
  - Identical in behavior to `f2_tx_mimo.m`, except adapted for the lower frequency.
  - Continuously alternates between two MIMO streams during transmission.

---

## General Notes
- All SDR hardware configurations (e.g., center frequency, gain, sample rate) must match the environment's requirements.
- Avoid overwriting existing `.mat` files by ensuring unique filenames.

---
