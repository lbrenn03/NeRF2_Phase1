# Data Processing for NeRF2 Phase 1

This directory contains scripts and resources to process RF signal data for creating NeRF2 training datasets. The workflow and scripts detailed below facilitate data preparation for machine learning models.

---

## Directory Contents

### Scripts

#### `setup.sh`
- **Purpose**: Bootstraps a Python virtual environment and installs necessary dependencies.
- **Usage**:
  1. Run the script to create a virtual environment and install dependencies:
     ```bash
     ./setup.sh
     ```
  2. Activate the virtual environment:
     ```bash
     source venv/bin/activate
     ```
     It's recommended to have Python 3.6 or higher installed.

- **Dependencies Managed**:
  - `numpy`, `scipy`, `matplotlib`, along with other optional packages in the `requirements.txt` file.

#### `signal_processing.py`
- **Purpose**: Processes `.mat` files containing RF signal data to extract relevant information critical for NeRF2 training.
- **Usage**:
  ```bash
  python signal_processing.py --input raw_capture.mat --output processed.mat
  ```
  Replace `raw_capture.mat` with the raw data file, and the processed file will be stored in the specified output path.

- **Key Outputs**:
  - Processes raw `.mat` files into structured datasets in the format `nerf_<tag>.mat`, matching MATLAB's final data struct layout for training purposes.

- **Important Parameters**:
  - Sampling rate and data capture time configured according to the MATLAB script used for transmission.

---

### Data Files

#### `requirements.txt`
- **Purpose**: Provides a list of Python dependencies required for running the scripts.
- **Dependencies**:
  - `numpy`
  - `scipy`
  - `matplotlib`
  - `scikit-learn`
  - `pyyaml`

---

## Workflow Overview

1. **Environment Setup**:
   - Run the `setup.sh` script to create and activate a virtual environment or ensure that the required Python dependencies (from `requirements.txt`) are installed.

2. **Prepare Input Data**:
   - Ensure `.mat` files generated from RF data collection scripts (e.g., MATLAB scripts) are correctly placed.

3. **Data Processing**:
   - Use the `signal_processing.py` script to process the captured `.mat` files into structured data for NeRF2 training.

4. **Inspect Outputs**:
   - Processed data in `.mat` format will be saved to the designated output path for use in machine learning workflows.

---

## Notes
- The default parameters of `signal_processing.py` are configured to align with the MATLAB scripts used for RF signal transmission.
- For any discrepancies or errors, verify the dependencies and configurations in `setup.sh` and `signal_processing.py`.
