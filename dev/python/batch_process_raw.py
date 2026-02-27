#!/usr/bin/env python3
"""
Batch Process .mat Files

This script processes all .mat files in an input directory using signal_processing.py.
Each file is processed and saved with a unique name in the output directory.

Usage:
    python batch_process_mat_files.py <input_dir> <output_dir> [--signal-processor <path>]

Example:
    python batch_process_mat_files.py ./raw_data ./processed_data
    python batch_process_mat_files.py ./raw_data ./processed_data --signal-processor ./signal_processing.py
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime


def find_mat_files(input_dir):
    """
    Find all .mat files in the input directory (non-recursive).
    
    Args:
        input_dir: Path to the input directory
        
    Returns:
        List of Path objects for .mat files
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
    
    # Find all .mat files (non-recursive)
    mat_files = list(input_path.glob("*.mat"))
    
    return sorted(mat_files)


def create_output_filename(input_file, output_dir):
    """
    Create a unique output filename based on the input filename.
    
    Args:
        input_file: Path object for the input file
        output_dir: Path to the output directory
        
    Returns:
        Path object for the output file
    """
    output_path = Path(output_dir)
    
    # Use the input filename stem (without extension) and add processed_ prefix
    base_name = input_file.stem
    output_name = f"processed_{base_name}.mat"
    
    output_file = output_path / output_name
    
    # If file already exists, add a timestamp to make it unique
    if output_file.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"processed_{base_name}_{timestamp}.mat"
        output_file = output_path / output_name
    
    return output_file


def process_file(input_file, output_file, signal_processor):
    """
    Process a single .mat file using signal_processing.py.
    
    Args:
        input_file: Path to the input .mat file
        output_file: Path to the output .mat file
        signal_processor: Path to the signal_processing.py script
        
    Returns:
        tuple: (success: bool, error_message: str or None)
    """
    try:
        # Build the command
        cmd = [
            sys.executable,  # Use the same Python interpreter
            str(signal_processor),
            "--input", str(input_file),
            "--output", str(output_file)
        ]
        
        # Run the command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            return True, None
        else:
            error_msg = result.stderr if result.stderr else result.stdout
            return False, error_msg
            
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Batch process .mat files using signal_processing.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_process_mat_files.py ./raw_data ./processed_data
  python batch_process_mat_files.py ./raw_data ./processed_data --signal-processor ./signal_processing.py
        """
    )
    
    parser.add_argument(
        "input_dir",
        help="Directory containing input .mat files"
    )
    
    parser.add_argument(
        "output_dir",
        help="Directory to save processed .mat files"
    )
    
    parser.add_argument(
        "--signal-processor",
        default="signal_processing.py",
        help="Path to signal_processing.py script (default: signal_processing.py in current directory)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed processing information"
    )
    
    args = parser.parse_args()
    
    # Validate signal processor exists
    signal_processor = Path(args.signal_processor)
    if not signal_processor.exists():
        print(f"ERROR: Signal processor not found: {args.signal_processor}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all .mat files
    try:
        mat_files = find_mat_files(args.input_dir)
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    if not mat_files:
        print(f"No .mat files found in {args.input_dir}")
        sys.exit(0)
    
    print(f"Found {len(mat_files)} .mat file(s) in {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Signal processor: {args.signal_processor}")
    print("-" * 60)
    
    # Process each file
    successful = 0
    failed = 0
    
    for i, input_file in enumerate(mat_files, 1):
        output_file = create_output_filename(input_file, output_dir)
        
        print(f"\n[{i}/{len(mat_files)}] Processing: {input_file.name}")
        if args.verbose:
            print(f"    Input:  {input_file}")
            print(f"    Output: {output_file}")
        
        success, error = process_file(input_file, output_file, signal_processor)
        
        if success:
            successful += 1
            print(f"    ✓ Success → {output_file.name}")
        else:
            failed += 1
            print(f"    ✗ Failed")
            if args.verbose and error:
                print(f"    Error: {error[:200]}")  # Print first 200 chars of error
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Processing complete:")
    print(f"  Successful: {successful}")
    print(f"  Failed:     {failed}")
    print(f"  Total:      {len(mat_files)}")
    print("=" * 60)
    
    # Exit with error code if any files failed
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()