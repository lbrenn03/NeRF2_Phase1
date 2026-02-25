#!/bin/bash
#
# setup_environment.sh
# Bootstrap script for NeRF2 RF dataset processing environment
#
# This script sets up a Python virtual environment and installs all required dependencies.
#

set -e  # Exit on any error

echo "=========================================="
echo "NeRF2 RF Dataset - Environment Setup"
echo "=========================================="
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.6 or higher and try again"
    exit 1
fi

# Display Python version
PYTHON_VERSION=$(python3 --version)
echo "Found: $PYTHON_VERSION"
echo ""

# Check if we're already in a virtual environment
if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "Already in a virtual environment: $VIRTUAL_ENV"
    echo "Proceeding with installation in current environment..."
    echo ""
else
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
        echo "✓ Virtual environment created"
        echo ""
    else
        echo "Virtual environment already exists"
        echo ""
    fi

    # Activate virtual environment
    echo "Activating virtual environment..."
    source venv/bin/activate
    echo "✓ Virtual environment activated"
    echo ""
fi

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip --quiet
echo "✓ pip upgraded"
echo ""

# Install requirements
echo "Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt --quiet
    echo "✓ Dependencies installed:"
    pip list | grep -E "numpy|scipy|matplotlib"
else
    echo "Warning: requirements.txt not found"
    echo "Installing dependencies manually..."
    pip install numpy scipy matplotlib --quiet
    echo "✓ Dependencies installed"
fi
echo ""

# Verify installation
echo "Verifying installation..."
python3 << 'EOF'
import sys
try:
    import numpy
    print(f"✓ numpy {numpy.__version__}")
except ImportError:
    print("✗ numpy not found")
    sys.exit(1)

try:
    import scipy
    print(f"✓ scipy {scipy.__version__}")
except ImportError:
    print("✗ scipy not found")
    sys.exit(1)

try:
    import matplotlib
    print(f"✓ matplotlib {matplotlib.__version__}")
except ImportError:
    print("✗ matplotlib not found")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Setup Complete!"
    echo "=========================================="
    echo ""
    echo "To use the environment:"
    echo "  1. Activate: source venv/bin/activate"
    echo "  2. Run script: python3 rssi_heatmap.py"
    echo "  3. Deactivate: deactivate"
    echo ""
else
    echo ""
    echo "Setup failed. Please check error messages above."
    exit 1
fi