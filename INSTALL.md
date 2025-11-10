# 🚀 Installation & Setup Guide

## Step-by-Step Installation

### Prerequisites
- Python 3.8 or higher
- 10GB+ free disk space
- (Optional but recommended) NVIDIA GPU with 8GB+ VRAM

---

## Option 1: Quick Install (Recommended)

### Windows (PowerShell)

```powershell
# 1. Check Python version
python --version

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify setup
python check_setup.py

# 4. Start training
python train.py
```

### Linux / macOS (Terminal)

```bash
# 1. Check Python version
python3 --version

# 2. Install dependencies
pip3 install -r requirements.txt

# 3. Verify setup
python3 check_setup.py

# 4. Start training
python3 train.py
```

---

## Option 2: Manual Install

### Step 1: Install Python

**Windows:**
1. Download Python 3.10+ from [python.org](https://www.python.org/downloads/)
2. Run installer
3. ✅ Check "Add Python to PATH"
4. Complete installation

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3 python3-pip
```

**macOS:**
```bash
brew install python@3.10
```

### Step 2: Install PyTorch

**With CUDA (GPU - Recommended):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**CPU Only:**
```bash
pip install torch torchvision
```

### Step 3: Install Other Dependencies

```bash
pip install gradio numpy scikit-image Pillow matplotlib tqdm
```

### Step 4: Verify Installation

```bash
python check_setup.py
```

Expected output:
```
✅ ALL CHECKS PASSED!
```

---

## Option 3: Conda Environment (Advanced)

### Create Environment

```bash
# Create new environment
conda create -n colorization python=3.10

# Activate environment
conda activate colorization

# Install PyTorch (GPU)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other packages
pip install gradio scikit-image tqdm
```

### Verify

```bash
python check_setup.py
```

---

## Troubleshooting

### Issue: "Python not found"

**Solution:**
- Ensure Python is in PATH
- Try `python3` instead of `python`
- Reinstall Python with PATH option checked

### Issue: "pip not found"

**Solution:**
```bash
# Windows
python -m ensurepip --upgrade

# Linux/macOS
sudo apt install python3-pip
```

### Issue: "CUDA not available"

**Check CUDA:**
```python
import torch
print(torch.cuda.is_available())  # Should be True
```

**Solutions:**
1. Install CUDA Toolkit from [NVIDIA](https://developer.nvidia.com/cuda-downloads)
2. Update GPU drivers
3. Reinstall PyTorch with CUDA:
   ```bash
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

### Issue: "Out of memory"

**Solution:**
In `colozier_gan.py` or `config.py`, reduce batch size:
```python
BATCH_SIZE = 16  # or 8
```

### Issue: Package version conflicts

**Solution:**
```bash
# Create fresh virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Install from requirements.txt
pip install -r requirements.txt
```

---

## Verify Installation

Run the setup checker:

```bash
python check_setup.py
```

### Expected Output:

```
======================================================================
IMAGE COLORIZATION - SYSTEM CHECK
======================================================================

Checking Python version...
  ✅ Python 3.10.12 (OK)

Checking required packages...
  ✅ PyTorch installed
  ✅ TorchVision installed
  ✅ Pillow installed
  ✅ NumPy installed
  ✅ scikit-image installed
  ✅ Matplotlib installed
  ✅ Gradio installed
  ✅ tqdm installed

Checking CUDA/GPU...
  ✅ CUDA available
  ✅ CUDA version: 11.8
  ✅ GPU count: 1
  ✅ GPU 0: NVIDIA GeForce RTX 3080
  ✅ GPU Memory: 10.0 GB
  ✅ Sufficient memory for batch size 32

Checking dataset...
  ✅ Dataset directory found
  ✅ Found 7129 images
  ✅ Large dataset - excellent!

Checking disk space...
  ℹ️  Free space: 150.5 GB
  ✅ Plenty of space for checkpoints

Checking project files...
  ✅ colozier_gan.py
  ✅ requirements.txt
  ✅ README.md
  ✅ QUICKSTART.md

======================================================================
SYSTEM CHECK SUMMARY
======================================================================

✅ ALL CHECKS PASSED!

Your system is ready for training.

Next steps:
1. Run: python train.py
2. Or uncomment training section in colozier_gan.py
3. Training will take 8-10 hours on GPU (RTX 3080)
4. Monitor progress via progress bars
5. Checkpoints saved every 10 epochs

======================================================================
```

---

## Hardware Requirements

### Minimum (Not Recommended)
- CPU: Any modern processor
- RAM: 8GB
- Storage: 10GB
- GPU: None (CPU training)
- Training Time: Days/weeks

### Recommended
- CPU: Intel i5/AMD Ryzen 5 or better
- RAM: 16GB
- Storage: 20GB SSD
- GPU: NVIDIA RTX 2060 or better (8GB VRAM)
- Training Time: 8-15 hours

### Optimal
- CPU: Intel i7/AMD Ryzen 7 or better
- RAM: 32GB
- Storage: 50GB NVMe SSD
- GPU: NVIDIA RTX 3080/4080 (10-16GB VRAM)
- Training Time: 4-8 hours

---

## Cloud Options (If No GPU)

### Google Colab (Free GPU)

1. Upload project to Google Drive
2. Open Google Colab
3. Enable GPU: Runtime → Change runtime type → GPU
4. Mount Drive and navigate to project
5. Run training cells

### Kaggle (Free GPU)

1. Create Kaggle account
2. Create new notebook
3. Enable GPU accelerator
4. Upload dataset and code
5. Run training

### AWS / Azure / GCP (Paid)

- AWS EC2: p3.2xlarge instance
- Azure: NC6 instance
- GCP: n1-standard-4 with V100

---

## Quick Start Commands

### 1. First-Time Setup
```bash
git clone <your-repo>  # or download zip
cd DLI_LAB_PROJECT
pip install -r requirements.txt
python check_setup.py
```

### 2. Start Training
```bash
python train.py
```

### 3. Run Inference (After Training)
```bash
python colozier_gan.py
```

---

## Package Versions (Tested)

```
Python: 3.8 - 3.11
torch: 2.0.0+
torchvision: 0.15.0+
gradio: 4.0.0+
numpy: 1.24.0+
scikit-image: 0.21.0+
Pillow: 10.0.0+
matplotlib: 3.7.0+
tqdm: 4.65.0+
```

---

## Common Installation Paths

### Windows
```
Python: C:\Users\<username>\AppData\Local\Programs\Python\Python310\
Pip: C:\Users\<username>\AppData\Local\Programs\Python\Python310\Scripts\
Project: C:\Users\<username>\Desktop\DLI_LAB_PROJECT\
```

### Linux
```
Python: /usr/bin/python3
Pip: /usr/bin/pip3
Project: ~/DLI_LAB_PROJECT/
```

### macOS
```
Python: /usr/local/bin/python3
Pip: /usr/local/bin/pip3
Project: ~/Desktop/DLI_LAB_PROJECT/
```

---

## Environment Variables (Optional)

Add to `.bashrc` or `.zshrc` (Linux/macOS):

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

Add to Environment Variables (Windows):
```
CUDA_PATH = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
```

---

## Next Steps After Installation

1. ✅ Run `python check_setup.py` to verify
2. ✅ Read `QUICKSTART.md` for usage guide
3. ✅ Review `README.md` for detailed docs
4. ✅ Start training with `python train.py`
5. ✅ Monitor progress in terminal
6. ✅ Wait for training to complete
7. ✅ Launch Gradio interface for inference

---

## Support & Resources

- **Documentation**: README.md
- **Quick Start**: QUICKSTART.md
- **Config**: config.py
- **PyTorch**: https://pytorch.org/
- **Gradio**: https://gradio.app/
- **CUDA**: https://developer.nvidia.com/cuda-toolkit

---

## Installation Checklist

- [ ] Python 3.8+ installed
- [ ] pip/pip3 working
- [ ] All packages installed
- [ ] CUDA working (if GPU)
- [ ] Dataset folder exists (GAN_color_images/)
- [ ] 10GB+ free disk space
- [ ] `check_setup.py` passes all tests
- [ ] Ready to start training!

---

**Having issues? Run `python check_setup.py` for diagnostics!**

**Ready to train? Run `python train.py`**

🎨 **Happy Colorizing!**
