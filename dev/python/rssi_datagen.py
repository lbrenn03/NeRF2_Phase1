#!/usr/bin/env python3
"""
Process .mat files to extract RSSI measurements and position data.

This script:
1. Reads all .mat files from a specified directory
2. Computes RSSI from all_syms_ant1 and all_syms_ant2
3. Generates gateway_rssi.csv with RSSI values per gateway (tx_pos + antenna combination)
4. Generates rx_pos.csv with receiver positions
5. Generates tx_pos.yaml with transmitter positions per gateway
"""

import os
import glob
import numpy as np
import scipy.io
import yaml
from pathlib import Path
from collections import defaultdict


def compute_rssi(symbols):
    """
    Compute RSSI in dBm from complex symbols.
    
    RSSI = 10 * log10(mean(|symbols|^2))
    
    Args:
        symbols: Complex-valued array of shape (n_samples, n_subcarriers)
    
    Returns:
        Array of RSSI values in dBm for each subcarrier
    """
    power = np.mean(np.abs(symbols)**2, axis=0)
    rssi = 10 * np.log10(power)
    return rssi


def load_mat_file(filepath):
    """
    Load a .mat file and extract relevant fields.
    
    Args:
        filepath: Path to the .mat file
    
    Returns:
        Dictionary with keys: all_syms_ant1, all_syms_ant2, rx_pos, tx_pos
        Returns None if file cannot be loaded
    """
    try:
        mat_data = scipy.io.loadmat(filepath)
        
        # Navigate the nested structure
        final_data = mat_data['final_data'][0, 0]
        
        # Extract fields
        data = {
            'all_syms_ant1': final_data['all_syms_ant1'],
            'all_syms_ant2': final_data['all_syms_ant2'],
            'rx_pos': final_data['rx_pos'].flatten(),  # Convert (1, 3) to (3,)
            'tx_pos': final_data['tx_pos'].flatten(),  # Convert (1, 3) to (3,)
        }
        
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def process_mat_files(data_dir):
    """
    Process all .mat files in the specified directory and subdirectories recursively.
    
    Args:
        data_dir: Directory containing .mat files
    
    Returns:
        Dictionary with:
            - rssi_data: List of (rx_idx, gateway_id, rssi_value) tuples
            - rx_positions: Dictionary mapping rx_idx to (x, y, z) position
            - tx_positions: Dictionary mapping gateway_id to (x, y, z) position
    """
    # Find all .mat files recursively
    mat_files = sorted(glob.glob(os.path.join(data_dir, '**', '*.mat'), recursive=True))
    
    if not mat_files:
        raise ValueError(f"No .mat files found in {data_dir}")
    
    print(f"Found {len(mat_files)} .mat files")
    
    # Track unique tx positions to assign gateway IDs
    tx_positions_list = []
    tx_pos_to_gateway = {}
    
    # Store RSSI data and positions
    rssi_measurements = []  # List of (rx_idx, gateway_id, rssi_value)
    rx_positions = {}  # Map rx_idx to position
    rx_idx = 0
    
    # Process each file
    for file_idx, mat_file in enumerate(mat_files):
        # Show relative path for cleaner output
        rel_path = os.path.relpath(mat_file, data_dir)
        print(f"Processing {rel_path} ({file_idx+1}/{len(mat_files)})...")
        
        data = load_mat_file(mat_file)
        if data is None:
            continue
        
        # Get tx position and assign gateway IDs
        tx_pos = tuple(data['tx_pos'])
        
        # Check if this tx position is new
        if tx_pos not in tx_pos_to_gateway:
            # Assign two gateway IDs for this tx position (one per antenna)
            base_gateway_num = len(tx_positions_list) * 2 + 1
            tx_pos_to_gateway[tx_pos] = {
                'ant1': f"g{base_gateway_num:02d}",
                'ant2': f"g{base_gateway_num+1:02d}"
            }
            tx_positions_list.append(tx_pos)
        
        gateway_ids = tx_pos_to_gateway[tx_pos]
        
        # Store rx position
        rx_positions[rx_idx] = data['rx_pos']
        
        # Compute RSSI for both antennas
        rssi_ant1 = compute_rssi(data['all_syms_ant1'])
        rssi_ant2 = compute_rssi(data['all_syms_ant2'])
        
        # Average across subcarriers to get single RSSI value per antenna
        rssi_ant1_avg = np.mean(rssi_ant1)
        rssi_ant2_avg = np.mean(rssi_ant2)
        
        # Store measurements
        rssi_measurements.append((rx_idx, gateway_ids['ant1'], rssi_ant1_avg))
        rssi_measurements.append((rx_idx, gateway_ids['ant2'], rssi_ant2_avg))
        
        rx_idx += 1
    
    # Build tx_positions dictionary with gateway IDs
    tx_positions = {}
    for tx_pos in tx_positions_list:
        gateway_ids = tx_pos_to_gateway[tx_pos]
        # Both antennas at same tx location
        tx_positions[gateway_ids['ant1']] = tx_pos
        tx_positions[gateway_ids['ant2']] = tx_pos
    
    return {
        'rssi_measurements': rssi_measurements,
        'rx_positions': rx_positions,
        'tx_positions': tx_positions
    }


def write_gateway_rssi_csv(rssi_measurements, rx_positions, tx_positions, output_file):
    """
    Write gateway_rssi.csv file.
    
    Format:
        g01,g02,g03,g04,g05,...
        -82.4,-75.5,-76.3,-67.0,-81.0,...
        -76.4,-77.6,-66.5,-74.2,-85.6,...
    """
    # Get sorted list of gateway IDs
    gateway_ids = sorted(tx_positions.keys(), key=lambda x: int(x[1:]))
    
    # Organize RSSI data by rx_idx and gateway
    rssi_matrix = defaultdict(dict)
    for rx_idx, gateway_id, rssi_value in rssi_measurements:
        rssi_matrix[rx_idx][gateway_id] = rssi_value
    
    # Write CSV
    with open(output_file, 'w') as f:
        # Header row
        f.write(','.join(gateway_ids) + '\n')
        
        # Data rows (one per rx position)
        for rx_idx in sorted(rx_positions.keys()):
            rssi_values = []
            for gateway_id in gateway_ids:
                rssi = rssi_matrix[rx_idx].get(gateway_id, -100.0)  # Default to -100 if missing
                rssi_values.append(f"{rssi:.1f}")
            f.write(','.join(rssi_values) + '\n')
    
    print(f"Written: {output_file}")


def write_rx_pos_csv(rx_positions, output_file):
    """
    Write rx_pos.csv file.
    
    Format:
        x, y, z
        16.13,20.55,0.11
        16.74,24.64,-0.20
        ...
    """
    with open(output_file, 'w') as f:
        # Header
        f.write('x, y, z\n')
        
        # Data rows
        for rx_idx in sorted(rx_positions.keys()):
            pos = rx_positions[rx_idx]
            f.write(f"{pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}\n")
    
    print(f"Written: {output_file}")


def write_tx_pos_yaml(tx_positions, output_file):
    """
    Write tx_pos.yaml file.
    
    Format:
        g01: [10.67, 25.63, 1.0]
        g02: [12.61, 24.87, 1.0]
        ...
    """
    # Convert to regular dict with sorted keys and formatted values
    tx_pos_dict = {}
    for gateway_id in sorted(tx_positions.keys(), key=lambda x: int(x[1:])):
        pos = tx_positions[gateway_id]
        # Format as list with 2 decimal places
        tx_pos_dict[gateway_id] = [round(float(pos[0]), 2), 
                                    round(float(pos[1]), 2), 
                                    round(float(pos[2]), 2)]
    
    # Write YAML
    with open(output_file, 'w') as f:
        yaml.dump(tx_pos_dict, f, default_flow_style=False, sort_keys=False)
    
    print(f"Written: {output_file}")


def main():
    """Main processing function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Process .mat files to generate RSSI and position data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python process_mat_files.py /path/to/data_dir
    python process_mat_files.py /path/to/data_dir --output-dir ./output
        """
    )
    parser.add_argument('data_dir', help='Directory containing .mat files')
    parser.add_argument('--output-dir', default=None, 
                        help='Output directory (default: same as data_dir)')
    
    args = parser.parse_args()
    
    # Validate input directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists() or not data_dir.is_dir():
        print(f"Error: {data_dir} is not a valid directory")
        return 1
    
    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else data_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Input directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print("="*60)
    
    # Process all .mat files
    try:
        results = process_mat_files(str(data_dir))
    except Exception as e:
        print(f"Error processing files: {e}")
        return 1
    
    print("="*60)
    print(f"Processed {len(results['rx_positions'])} RX positions")
    print(f"Found {len(results['tx_positions']) // 2} unique TX positions")
    print(f"Generated {len(results['tx_positions'])} gateways (2 antennas per TX)")
    print("="*60)
    
    # Write output files
    gateway_rssi_file = output_dir / 'gateway_rssi.csv'
    rx_pos_file = output_dir / 'rx_pos.csv'
    tx_pos_file = output_dir / 'tx_pos.yaml'
    
    write_gateway_rssi_csv(results['rssi_measurements'], 
                           results['rx_positions'],
                           results['tx_positions'],
                           str(gateway_rssi_file))
    
    write_rx_pos_csv(results['rx_positions'], str(rx_pos_file))
    
    write_tx_pos_yaml(results['tx_positions'], str(tx_pos_file))
    
    print("="*60)
    print("Processing complete!")
    
    return 0


if __name__ == '__main__':
    exit(main())