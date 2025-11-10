#!/bin/bash
# ===================================================================
# Image Colorization - Linux/macOS Setup & Training Launcher
# ===================================================================

echo ""
echo "===================================================================="
echo "IMAGE COLORIZATION - SETUP AND TRAINING"
echo "===================================================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 is not installed"
    echo ""
    echo "Please install Python 3.8+ first:"
    echo "  Ubuntu/Debian: sudo apt install python3 python3-pip"
    echo "  macOS: brew install python@3.10"
    echo ""
    exit 1
fi

echo "[OK] Python is installed"
PYTHON_VERSION=$(python3 --version)
echo "    $PYTHON_VERSION"
echo ""

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "[ERROR] pip3 is not installed"
    echo ""
    echo "Please install pip3:"
    echo "  Ubuntu/Debian: sudo apt install python3-pip"
    echo "  macOS: python3 -m ensurepip"
    echo ""
    exit 1
fi

echo "[OK] pip is installed"
echo ""

# Check if requirements are installed
echo "Checking dependencies..."
python3 -c "import torch" &> /dev/null
if [ $? -ne 0 ]; then
    echo ""
    echo "[INFO] Dependencies not installed. Installing now..."
    echo ""
    echo "This may take 5-10 minutes..."
    echo ""
    
    pip3 install -r requirements.txt
    
    if [ $? -ne 0 ]; then
        echo ""
        echo "[ERROR] Failed to install dependencies"
        echo ""
        echo "Try manually: pip3 install -r requirements.txt"
        echo ""
        exit 1
    fi
    
    echo ""
    echo "[OK] Dependencies installed successfully"
else
    echo "[OK] Dependencies already installed"
fi

echo ""
echo "===================================================================="
echo "Running system check..."
echo "===================================================================="
echo ""

# Run system check
python3 check_setup.py
CHECK_RESULT=$?

if [ $CHECK_RESULT -ne 0 ]; then
    echo ""
    echo "[WARNING] Some checks failed. Review the output above."
    echo ""
    read -p "Do you want to continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "Setup cancelled."
        exit 1
    fi
fi

echo ""
echo "===================================================================="
echo "Starting training..."
echo "===================================================================="
echo ""
echo "This will take approximately 8-10 hours on GPU (RTX 3080)"
echo "You can press Ctrl+C to stop training at any time"
echo ""
read -p "Press Enter to start training, or Ctrl+C to cancel..."

# Start training
python3 train.py

echo ""
echo "===================================================================="
echo "Training completed or stopped"
echo "===================================================================="
echo ""
