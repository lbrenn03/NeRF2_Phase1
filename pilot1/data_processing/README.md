# Data Processing for NeRF2 Phase 1

This directory contains scripts and resources to process RF signal data for creating NeRF2 training datasets. The workflow and scripts detailed below facilitate data preparation for machine learning.

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
- **Purpose**: Processes a single `.mat` file containing RF signal data into structured output for NeRF2 training.
- **Usage**:
  ```bash
  python signal_processing.py --input raw_capture.mat --output processed.mat
  ```
  Replace `raw_capture.mat` with the raw data file, and the processed file will be stored in the specified output location.

#### `batch_process_raw.py`
- **Purpose**: Automates batch processing of multiple `.mat` files by executing `signal_processing.py` for each input file.
- **Usage**:
  ```bash
  python batch_process_raw.py <input_dir> <output_dir>
  ```
  - Replace `<input_dir>` with the folder containing `.mat` files to process.
  - Replace `<output_dir>` with the destination folder for processed `.mat` files.
  - Optional: Specify the location of `signal_processing.py` using `--signal-processor`.
  - Use `--verbose` or `-v` for detailed processing logs.

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

## High-Level Usage Guide

1. **Environment Setup**:
   - Start by running `setup.sh` to create a Python virtual environment and install dependencies necessary to run the scripts.

2. **Input Data Preparation**:
   - Place `.mat` files (generated from RF data collection using MATLAB scripts) into the `raw_data/` directory or another folder of your choice.

3. **Data Processing**:
   - Use `signal_processing.py` for processing individual files or `batch_process_raw.py` for processing multiple files in bulk.

4. **Inspect Processed Data**:
   - Review the `.mat` files saved in the output directory to ensure they meet the requirements for NeRF2 training.

5. **Integrate Processed Data**:
   - The processed datasets are ready for use in downstream NeRF2 workflows, such as training or evaluation.

---

## Workflow Overview

1. **Environment Setup**:
   - Run the `setup.sh` script to create and activate a virtual environment or ensure that the required Python dependencies (from `requirements.txt`) are installed.

2. **Prepare Input Data**:
   - Ensure `.mat` files generated from RF data collection scripts (e.g., MATLAB scripts) are correctly placed.

3. **Data Processing**:
   - For a single file:
     ```bash
     python signal_processing.py --input raw_capture.mat --output processed_capture.mat
     ```
   - For batch processing:
     ```bash
     python batch_process_raw.py raw_data/ processed_data/
     ```
     Replace `raw_data/` with the folder containing raw `.mat` files. The processed files will be saved in `processed_data/`.

4. **Inspect Outputs**:
   - Processed data in `.mat` format will be saved to the designated output path.

---

## Notes
- The processing pipeline relies on `signal_processing.py`, including options for running without modification or through the batch script.
- For any discrepancies or errors, verify the dependencies and configurations in `setup.sh`, `signal_processing.py`, and `batch_process_raw.py`.